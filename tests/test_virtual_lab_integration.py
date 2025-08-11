"""
Comprehensive test suite for Virtual Lab Integration

This module tests the Virtual Lab integration functionality to ensure
all components work correctly and the integration is not faked.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Now try to import Virtual Lab components
try:
    from core.virtual_lab_integration.agent import Agent as VirtualLabAgent
    from core.virtual_lab_integration import Agent, __version__
    IMPORTS_SUCCESSFUL = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = str(e)


class TestVirtualLabAgent(unittest.TestCase):
    """Test the Virtual Lab Agent class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest(f"Import failed: {IMPORT_ERROR}")
            
        self.agent = VirtualLabAgent(
            title="Test Researcher",
            expertise="Machine Learning",
            goal="conduct experiments",
            role="research scientist",
            model="gpt-4"
        )
    
    def test_agent_initialization(self):
        """Test agent initialization with all required parameters."""
        self.assertEqual(self.agent.title, "Test Researcher")
        self.assertEqual(self.agent.expertise, "Machine Learning")
        self.assertEqual(self.agent.goal, "conduct experiments")
        self.assertEqual(self.agent.role, "research scientist")
        self.assertEqual(self.agent.model, "gpt-4")
    
    def test_agent_prompt_generation(self):
        """Test that agent generates proper prompt string."""
        expected_prompt = (
            "You are a Test Researcher. "
            "Your expertise is in Machine Learning. "
            "Your goal is to conduct experiments. "
            "Your role is to be a research scientist."
        )
        self.assertEqual(self.agent.prompt, expected_prompt)
    
    def test_agent_message_format(self):
        """Test that agent generates proper OpenAI API message format."""
        message = self.agent.message
        self.assertIsInstance(message, dict)
        self.assertEqual(message["role"], "system")
        self.assertEqual(message["content"], self.agent.prompt)
    
    def test_agent_hash_function(self):
        """Test that agent hash function works correctly."""
        self.assertIsInstance(hash(self.agent), int)
    
    def test_agent_equality(self):
        """Test that agent equality comparison works correctly."""
        agent1 = VirtualLabAgent(
            title="Test Researcher",
            expertise="Machine Learning",
            goal="conduct experiments",
            role="research scientist",
            model="gpt-4"
        )
        agent2 = VirtualLabAgent(
            title="Test Researcher",
            expertise="Machine Learning",
            goal="conduct experiments",
            role="research scientist",
            model="gpt-4"
        )
        agent3 = VirtualLabAgent(
            title="Different Researcher",
            expertise="Machine Learning",
            goal="conduct experiments",
            role="research scientist",
            model="gpt-4"
        )
        
        self.assertEqual(agent1, agent2)
        self.assertNotEqual(agent1, agent3)
        self.assertNotEqual(agent1, "not an agent")
    
    def test_agent_string_representations(self):
        """Test that agent string representations work correctly."""
        self.assertIn("Test Researcher", str(self.agent))
        self.assertIn("Test Researcher", repr(self.agent))


class TestVirtualLabIntegration(unittest.TestCase):
    """Test Virtual Lab integration functionality."""
    
    def test_module_imports_successfully(self):
        """Test that Virtual Lab module imports without errors."""
        if not IMPORTS_SUCCESSFUL:
            self.fail(f"Import failed: {IMPORT_ERROR}")
        
        # Test that we can create an agent
        agent = VirtualLabAgent(
            title="Test Agent",
            expertise="Testing",
            goal="test functionality",
            role="tester",
            model="gpt-4"
        )
        self.assertIsInstance(agent, VirtualLabAgent)
    
    def test_constants_available(self):
        """Test that Virtual Lab constants are available."""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest(f"Import failed: {IMPORT_ERROR}")
        
        # Test that version is available
        self.assertIsInstance(__version__, str)
        self.assertGreater(len(__version__), 0)
    
    def test_prompts_available(self):
        """Test that Virtual Lab prompts are available."""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest(f"Import failed: {IMPORT_ERROR}")
        
        # Test that we can create an agent with prompts
        agent = VirtualLabAgent(
            title="Test Agent",
            expertise="Testing",
            goal="test functionality",
            role="tester",
            model="gpt-4"
        )
        
        # Test that prompt is generated correctly
        self.assertIsInstance(agent.prompt, str)
        self.assertGreater(len(agent.prompt), 0)
        self.assertIn("Test Agent", agent.prompt)


class TestVirtualLabEnhanced(unittest.TestCase):
    """Test Enhanced Virtual Lab functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest(f"Import failed: {IMPORT_ERROR}")
    
    @patch('core.virtual_lab_enhanced.OpenAI')
    @patch('core.virtual_lab_enhanced.tqdm')
    def test_enhanced_virtual_lab_imports(self, mock_tqdm, mock_openai):
        """Test that enhanced virtual lab imports work correctly."""
        # This test uses mocks to avoid actual API calls
        try:
            from core.virtual_lab_enhanced import EnhancedVirtualLabMeetingSystem
            self.assertTrue(True)  # Import successful
        except ImportError as e:
            self.fail(f"Enhanced Virtual Lab import failed: {e}")
    
    def test_virtual_lab_agent_conversion(self):
        """Test conversion between Virtual Lab and framework agents."""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest(f"Import failed: {IMPORT_ERROR}")
        
        # Create a Virtual Lab agent
        vl_agent = VirtualLabAgent(
            title="Test Researcher",
            expertise="Machine Learning",
            goal="conduct experiments",
            role="research scientist",
            model="gpt-4"
        )
        
        # Test that we can access its properties
        self.assertEqual(vl_agent.title, "Test Researcher")
        self.assertEqual(vl_agent.expertise, "Machine Learning")
        self.assertEqual(vl_agent.goal, "conduct experiments")
        self.assertEqual(vl_agent.role, "research scientist")
        self.assertEqual(vl_agent.model, "gpt-4")
        
        # Test that prompt is generated correctly
        expected_prompt = (
            "You are a Test Researcher. "
            "Your expertise is in Machine Learning. "
            "Your goal is to conduct experiments. "
            "Your role is to be a research scientist."
        )
        self.assertEqual(vl_agent.prompt, expected_prompt)


class TestSystemIntegration(unittest.TestCase):
    """Test system-level integration."""
    
    def test_import_chain_validation(self):
        """Test that the import chain works correctly."""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest(f"Import failed: {IMPORT_ERROR}")
        
        # Test that we can import and use the agent
        agent = VirtualLabAgent(
            title="System Test Agent",
            expertise="System Testing",
            goal="validate system integration",
            role="system tester",
            model="gpt-4"
        )
        
        # Test basic functionality
        self.assertIsInstance(agent, VirtualLabAgent)
        self.assertIsInstance(agent.prompt, str)
        self.assertIsInstance(agent.message, dict)
        self.assertIsInstance(hash(agent), int)
    
    def test_agent_system_compatibility(self):
        """Test that agents are compatible with the system."""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest(f"Import failed: {IMPORT_ERROR}")
        
        # Test multiple agents
        agents = [
            VirtualLabAgent(
                title=f"Agent {i}",
                expertise=f"Expertise {i}",
                goal=f"Goal {i}",
                role=f"Role {i}",
                model="gpt-4"
            )
            for i in range(3)
        ]
        
        # Test that all agents work correctly
        for i, agent in enumerate(agents):
            self.assertEqual(agent.title, f"Agent {i}")
            self.assertEqual(agent.expertise, f"Expertise {i}")
            self.assertEqual(agent.goal, f"Goal {i}")
            self.assertEqual(agent.role, f"Role {i}")
            self.assertIsInstance(agent.prompt, str)
            self.assertGreater(len(agent.prompt), 0)

    def test_team_member_prompt_includes_agenda(self):
        """Test that team member prompts include agenda content."""
        from core.virtual_lab_integration.prompts import team_meeting_team_member_prompt
        from core.virtual_lab_integration.agent import Agent
        
        # Create a test agent
        test_agent = Agent(
            title="Test Expert",
            expertise="Test Domain",
            goal="test goals",
            role="test role",
            model="gpt-4"
        )
        
        # Test with agenda
        test_agenda = "Research novel prompting techniques"
        prompt_with_agenda = team_meeting_team_member_prompt(
            team_member=test_agent,
            round_num=1,
            num_rounds=3,
            agenda=test_agenda
        )
        
        # Verify agenda is included in prompt
        self.assertIn(test_agenda, prompt_with_agenda)
        self.assertIn("Agenda:", prompt_with_agenda)
        
        # Test without agenda (should still work)
        prompt_without_agenda = team_meeting_team_member_prompt(
            team_member=test_agent,
            round_num=1,
            num_rounds=3
        )
        
        # Verify prompt still works without agenda
        self.assertIn("please provide your thoughts", prompt_without_agenda)
        self.assertNotIn("Agenda:", prompt_without_agenda)


if __name__ == '__main__':
    unittest.main()