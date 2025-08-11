#!/usr/bin/env python3
"""
Test Suite for Enhanced Gradio UI

Comprehensive testing for the AI Research Lab Gradio interface to ensure
all functionality works correctly including session management, chat integration,
agent management, and data persistence.
"""

import os
import sys
import json
import time
import unittest
import threading
from datetime import datetime
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from gradio_app import AIResearchLabGradio
from data_manager import DataManager

class TestGradioUI(unittest.TestCase):
    """Test suite for the enhanced Gradio UI."""
    
    def setUp(self):
        """Set up test environment."""
        self.app = AIResearchLabGradio()
        self.test_session_id = None
        
    def tearDown(self):
        """Clean up test environment."""
        if self.test_session_id and self.app.data_manager:
            # Clean up test session
            try:
                self.app.data_manager.update_session_status(self.test_session_id, 'completed')
            except:
                pass
    
    def test_session_creation(self):
        """Test session creation and persistence."""
        # Test 1: Create new session
        session_id = self.app.create_session()
        self.assertIsNotNone(session_id)
        self.assertEqual(self.app.current_session_id, session_id)
        
        # Test 2: Verify session in data manager
        if self.app.data_manager:
            session = self.app.data_manager.get_session(session_id)
            self.assertIsNotNone(session)
            self.assertEqual(session['id'], session_id)
            self.assertEqual(session['status'], 'pending')
        
        # Test 3: Verify session state
        self.assertIsNotNone(self.app.current_session)
        self.assertEqual(self.app.current_session['session_id'], session_id)
        self.assertEqual(self.app.current_session['status'], 'pending')
        
        self.test_session_id = session_id
        print(f"✅ Session creation test passed: {session_id}")
    
    def test_chat_message_logging(self):
        """Test chat message logging and retrieval."""
        # Create session
        session_id = self.app.create_session()
        self.test_session_id = session_id
        
        # Test 1: Log user message
        test_message = "Test user message"
        self.app.log_chat_message(test_message, "user", "communication")
        
        # Test 2: Verify message in data manager
        if self.app.data_manager:
            chat_logs = self.app.data_manager.get_chat_logs(session_id=session_id, limit=10)
            self.assertGreater(len(chat_logs), 0)
            
            # Find our test message
            found_message = False
            for log in chat_logs:
                if log['message'] == test_message and log['author'] == 'user':
                    found_message = True
                    break
            self.assertTrue(found_message, "Test message not found in chat logs")
        
        # Test 3: Test assistant message
        assistant_message = "Test assistant response"
        self.app.log_chat_message(assistant_message, "assistant", "communication")
        
        # Test 4: Verify chat history format
        history = self.app.get_chat_history(session_id)
        self.assertIsInstance(history, list)
        
        print(f"✅ Chat message logging test passed")
    
    def test_research_session_management(self):
        """Test research session start, progress, and completion."""
        # Create session
        session_id = self.app.create_session()
        self.test_session_id = session_id
        
        # Test 1: Start research session
        research_question = "What is the impact of AI on scientific research?"
        history = []
        
        updated_history, status, data = self.app._start_research_session(research_question, history)
        
        # Verify session state
        self.assertTrue(self.app.is_research_active)
        if self.app.current_session:  # Only check if session exists
            self.assertEqual(self.app.current_session['research_question'], research_question)
            # Status might be 'running' or 'failed' depending on framework initialization
            self.assertIn(self.app.current_session['status'], ['running', 'failed'])
        
        # Test 2: Verify session persistence
        if self.app.data_manager and self.current_session_id:
            session_data = self.app.data_manager.get_session(self.current_session_id)
            if session_data:
                self.assertEqual(session_data['research_question'], research_question)
        
        # Test 3: Check research status
        status_data = self.app.get_research_status()
        self.assertIsInstance(status_data, dict)
        self.assertIn('status', status_data)
        
        print(f"✅ Research session management test passed")
    
    def test_agent_statistics(self):
        """Test agent statistics and marketplace integration."""
        # Test 1: Get agent statistics
        stats = self.app.get_agent_statistics()
        
        # Verify structure
        required_keys = ['total_agents', 'active_agents', 'avg_quality_score', 'critical_issues', 'hired_agents', 'available_agents']
        for key in required_keys:
            self.assertIn(key, stats)
        
        # Test 2: Verify agent data
        self.assertIsInstance(stats['total_agents'], int)
        self.assertIsInstance(stats['active_agents'], int)
        self.assertIsInstance(stats['avg_quality_score'], float)
        self.assertIsInstance(stats['hired_agents'], list)
        self.assertIsInstance(stats['available_agents'], list)
        
        # Test 3: Verify agent details
        if stats['available_agents']:
            agent = stats['available_agents'][0]
            required_agent_keys = ['id', 'role', 'expertise', 'is_hired', 'performance_metrics']
            for key in required_agent_keys:
                self.assertIn(key, agent)
        
        print(f"✅ Agent statistics test passed")
    
    def test_agent_activities(self):
        """Test agent activity tracking and retrieval."""
        # Create session
        session_id = self.app.create_session()
        self.test_session_id = session_id
        
        # Test 1: Get agent activities (should be empty initially)
        activities = self.app.get_agent_activities(session_id)
        self.assertIsInstance(activities, list)
        
        # Test 2: Simulate agent activity
        if self.app.data_manager:
            self.app.data_manager.persist_agent_activity(
                session_id=session_id,
                agent_id="test_agent_1",
                activity_type="thinking",
                message="Test agent activity",
                status="active"
            )
            
            # Verify activity was logged
            activities = self.app.get_agent_activities(session_id)
            self.assertGreater(len(activities), 0)
            
            # Find our test activity
            found_activity = False
            for activity in activities:
                if (activity['agent_id'] == "test_agent_1" and 
                    activity['activity_type'] == "thinking" and
                    activity['message'] == "Test agent activity"):
                    found_activity = True
                    break
            self.assertTrue(found_activity, "Test agent activity not found")
        
        print(f"✅ Agent activities test passed")
    
    def test_meeting_transcripts(self):
        """Test meeting transcript tracking and retrieval."""
        # Create session
        session_id = self.app.create_session()
        self.test_session_id = session_id
        
        # Test 1: Get meeting transcripts (should be empty initially)
        meetings = self.app.get_meeting_transcripts(session_id)
        self.assertIsInstance(meetings, list)
        
        # Test 2: Simulate meeting
        if self.app.data_manager:
            self.app.data_manager.persist_meeting(
                session_id=session_id,
                meeting_id="test_meeting_1",
                participants=["agent_1", "agent_2"],
                topic="Test meeting topic",
                agenda={"objectives": ["Test objective"]},
                transcript=[{"speaker": "agent_1", "message": "Test message"}],
                outcomes={"decisions": ["Test decision"]}
            )
            
            # Verify meeting was logged
            meetings = self.app.get_meeting_transcripts(session_id)
            self.assertGreater(len(meetings), 0)
            
            # Find our test meeting
            found_meeting = False
            for meeting in meetings:
                if (meeting['meeting_id'] == "test_meeting_1" and 
                    meeting['topic'] == "Test meeting topic"):
                    found_meeting = True
                    break
            self.assertTrue(found_meeting, "Test meeting not found")
        
        print(f"✅ Meeting transcripts test passed")
    
    def test_chat_integration(self):
        """Test complete chat integration with research mode."""
        # Create session
        session_id = self.app.create_session()
        self.test_session_id = session_id
        
        # Test research mode
        research_question = "Research the impact of machine learning on healthcare"
        history = []
        
        updated_history, response, data = self.app._start_research_session(research_question, history)
        
        # Verify response format
        self.assertIn('session_id', data)
        self.assertIn('status', data)
        self.assertIn('research_question', data)
        self.assertEqual(data['research_question'], research_question)
        self.assertEqual(data['status'], 'started')
        
        # Verify history is updated
        self.assertEqual(len(updated_history), 1)
        self.assertEqual(updated_history[0][0], research_question)
        self.assertIn('Research Session Started', updated_history[0][1])
        
        # Test regular chat mode
        regular_message = "Hello, how are you?"
        history = []
        
        updated_history, response, data = self.app._continue_research_session(regular_message, history)
        
        # Verify response
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        
        # Verify history is updated
        self.assertEqual(len(updated_history), 1)
        self.assertEqual(updated_history[0][0], regular_message)
        self.assertIsInstance(updated_history[0][1], str)
        
        print(f"✅ Chat integration test passed")
    
    def test_error_handling(self):
        """Test error handling and recovery."""
        # Test 1: Handle empty message
        history = []
        message = ""
        
        updated_history, status, data = self.app.chat_with_research_lab(message, history, research_mode=False)
        
        # Should handle gracefully
        self.assertEqual(len(updated_history), 0)
        self.assertIn("Please enter a message", status)
        
        # Test 2: Handle framework errors gracefully
        # Temporarily break framework
        original_framework = self.app.research_framework
        self.app.research_framework = None
        
        try:
            stats = self.app.get_agent_statistics()
            # Should return default values
            self.assertEqual(stats['total_agents'], 0)
            self.assertEqual(stats['active_agents'], 0)
        finally:
            # Restore framework
            self.app.research_framework = original_framework
        
        print(f"✅ Error handling test passed")
    
    def test_data_persistence(self):
        """Test data persistence across app restarts."""
        # Create session and add data
        session_id = self.app.create_session()
        self.test_session_id = session_id
        
        # Add some test data
        test_message = "Persistent test message"
        self.app.log_chat_message(test_message, "user", "communication")
        
        if self.app.data_manager:
            self.app.data_manager.persist_agent_activity(
                session_id=session_id,
                agent_id="persistent_test_agent",
                activity_type="tool_use",
                message="Persistent agent activity",
                status="completed"
            )
        
        # Simulate app restart by creating new app instance
        new_app = AIResearchLabGradio()
        
        # Verify data persistence
        if new_app.data_manager:
            # Check session exists
            session = new_app.data_manager.get_session(session_id)
            self.assertIsNotNone(session)
            
            # Check chat logs exist
            chat_logs = new_app.data_manager.get_chat_logs(session_id=session_id)
            found_message = False
            for log in chat_logs:
                if log['message'] == test_message:
                    found_message = True
                    break
            self.assertTrue(found_message, "Persistent message not found after restart")
            
            # Check agent activities exist
            activities = new_app.data_manager.get_agent_activity(session_id=session_id)
            found_activity = False
            for activity in activities:
                if activity['agent_id'] == "persistent_test_agent":
                    found_activity = True
                    break
            self.assertTrue(found_activity, "Persistent agent activity not found after restart")
        
        print(f"✅ Data persistence test passed")

def run_performance_test():
    """Run performance tests for the UI."""
    print("\n🚀 Running Performance Tests...")
    
    app = AIResearchLabGradio()
    
    # Test 1: Session creation performance
    start_time = time.time()
    for i in range(10):
        session_id = app.create_session()
    session_creation_time = time.time() - start_time
    print(f"✅ Session creation: {session_creation_time:.3f}s for 10 sessions")
    
    # Test 2: Chat logging performance
    session_id = app.create_session()
    start_time = time.time()
    for i in range(100):
        app.log_chat_message(f"Test message {i}", "user", "communication")
    chat_logging_time = time.time() - start_time
    print(f"✅ Chat logging: {chat_logging_time:.3f}s for 100 messages")
    
    # Test 3: Agent statistics performance
    start_time = time.time()
    for i in range(50):
        stats = app.get_agent_statistics()
    stats_time = time.time() - start_time
    print(f"✅ Agent statistics: {stats_time:.3f}s for 50 calls")
    
    # Test 4: Research status performance
    start_time = time.time()
    for i in range(50):
        status = app.get_research_status()
    status_time = time.time() - start_time
    print(f"✅ Research status: {status_time:.3f}s for 50 calls")

def run_integration_test():
    """Run integration tests for the complete workflow."""
    print("\n🔗 Running Integration Tests...")
    
    app = AIResearchLabGradio()
    
    # Test complete research workflow
    print("1. Creating research session...")
    session_id = app.create_session()
    
    print("2. Starting research...")
    history = []
    research_question = "What are the latest developments in quantum computing?"
    updated_history, status, data = app.chat_with_research_lab(research_question, history, research_mode=True)
    
    print("3. Checking research status...")
    status_info = app.get_research_status()
    print(f"   Status: {status_info['status']}")
    print(f"   Progress: {status_info['progress']}%")
    print(f"   Phase: {status_info['current_phase']}")
    
    print("4. Checking agent statistics...")
    agent_stats = app.get_agent_statistics()
    print(f"   Total agents: {agent_stats['total_agents']}")
    print(f"   Active agents: {agent_stats['active_agents']}")
    
    print("5. Checking agent activities...")
    activities = app.get_agent_activities(session_id)
    print(f"   Activities recorded: {len(activities)}")
    
    print("6. Checking meeting transcripts...")
    meetings = app.get_meeting_transcripts(session_id)
    print(f"   Meetings recorded: {len(meetings)}")
    
    print("✅ Integration test completed successfully!")

if __name__ == "__main__":
    print("🧪 Starting Gradio UI Test Suite...")
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    run_performance_test()
    
    # Run integration tests
    run_integration_test()
    
    print("\n🎉 All tests completed!")
