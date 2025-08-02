#!/usr/bin/env python3
"""
Simple test script to verify web UI functionality
"""

import requests
import json
import time

def test_web_ui():
    base_url = "http://localhost:5000"
    
    print("Testing Web UI functionality...")
    
    # Test 1: Check if server is running
    try:
        response = requests.get(f"{base_url}/api/config")
        if response.status_code == 200:
            print("✓ Server is running")
        else:
            print("✗ Server not responding")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        return False
    
    # Test 2: Test chat logs functionality
    try:
        # Add a test chat log
        chat_data = {
            "session_id": "test_session",
            "type": "thought",
            "author": "Test Agent",
            "message": "This is a test thought"
        }
        response = requests.post(f"{base_url}/api/chat-logs", json=chat_data)
        if response.status_code == 200:
            print("✓ Chat logs POST endpoint working")
        else:
            print(f"✗ Chat logs POST failed: {response.status_code}")
        
        # Get chat logs
        response = requests.get(f"{base_url}/api/chat-logs")
        if response.status_code == 200:
            data = response.json()
            if data.get('logs'):
                print("✓ Chat logs GET endpoint working")
            else:
                print("✗ No chat logs returned")
        else:
            print(f"✗ Chat logs GET failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Chat logs test failed: {e}")
    
    # Test 3: Test agent activity functionality
    try:
        # Add test agent activity
        activity_data = {
            "session_id": "test_session",
            "agent_id": "test_agent",
            "type": "thinking",
            "message": "Test agent activity",
            "status": "active"
        }
        response = requests.post(f"{base_url}/api/agent-activity", json=activity_data)
        if response.status_code == 200:
            print("✓ Agent activity POST endpoint working")
        else:
            print(f"✗ Agent activity POST failed: {response.status_code}")
        
        # Get agent activity
        response = requests.get(f"{base_url}/api/agent-activity")
        if response.status_code == 200:
            data = response.json()
            if data.get('activities'):
                print("✓ Agent activity GET endpoint working")
            else:
                print("✗ No agent activities returned")
        else:
            print(f"✗ Agent activity GET failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Agent activity test failed: {e}")
    
    # Test 4: Test meetings functionality
    try:
        # Add test meeting
        meeting_data = {
            "session_id": "test_session",
            "meeting_id": "test_meeting",
            "participants": '["Agent 1", "Agent 2"]',
            "topic": "Test Meeting",
            "duration": 15,
            "outcome": "Test outcome"
        }
        response = requests.post(f"{base_url}/api/meetings", json=meeting_data)
        if response.status_code == 200:
            print("✓ Meetings POST endpoint working")
        else:
            print(f"✗ Meetings POST failed: {response.status_code}")
        
        # Get meetings
        response = requests.get(f"{base_url}/api/meetings")
        if response.status_code == 200:
            data = response.json()
            if data.get('meetings'):
                print("✓ Meetings GET endpoint working")
            else:
                print("✗ No meetings returned")
        else:
            print(f"✗ Meetings GET failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Meetings test failed: {e}")
    
    # Test 5: Test main page
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200 and "AI Research Lab" in response.text:
            print("✓ Main page loading correctly")
        else:
            print("✗ Main page not loading correctly")
    except Exception as e:
        print(f"✗ Main page test failed: {e}")
    
    print("\nWeb UI test completed!")
    return True

if __name__ == "__main__":
    test_web_ui() 