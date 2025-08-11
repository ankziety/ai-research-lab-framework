"""
Virtual Lab Meeting System Integration

Integrates the Virtual Lab based meeting methodology with the current AI researcher framework.
This system enables long-running, persistent meetings using OpenAI Assistants API while
preserving the existing agent marketplace and researcher capabilities.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

from openai import OpenAI

from agents.virtual_lab_bridge import VirtualLabAgentBridge
from agents.base_agent import BaseAgent
from agents.agent_marketplace import AgentMarketplace
from core.virtual_lab_integration.run_meeting import run_meeting
from core.virtual_lab_integration.constants import CONSISTENT_TEMPERATURE

logger = logging.getLogger(__name__)


class VirtualLabMeetingSystem:
    """
    Enhanced meeting system using Virtual Lab methodology (Swanson et al. 2023).
    
    This class provides a bridge between the current agent system and the Virtual Lab
    meeting methodology (Swanson et al. 2023), enabling long-running, persistent research sessions.
    """
    
    def __init__(self, pi_agent: BaseAgent, scientific_critic: BaseAgent, 
                 agent_marketplace: AgentMarketplace, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Virtual Lab meeting system.
        
        Args:
            pi_agent: Principal Investigator agent
            scientific_critic: Scientific Critic agent
            agent_marketplace: Agent marketplace for hiring
            config: Configuration dictionary
        """
        self.pi_agent = pi_agent
        self.scientific_critic = scientific_critic
        self.agent_marketplace = agent_marketplace
        self.config = config or {}
        
        # Initialize OpenAI client
        self.client = OpenAI()
        
        # Initialize agent bridges
        self.pi_bridge = VirtualLabAgentBridge(pi_agent)
        self.critic_bridge = VirtualLabAgentBridge(scientific_critic)
        
        # Initialize assistants for persistent state
        self._initialize_assistants()
        
        # Session tracking
        self.current_session_id = None
        self.meeting_history = []
        self.active_meetings = {}
        
        logger.info("Virtual Lab Meeting System initialized")
    
    def _initialize_assistants(self):
        """Initialize OpenAI assistants for PI and Scientific Critic."""
        try:
            self.pi_bridge.initialize_assistant(self.client)
            self.critic_bridge.initialize_assistant(self.client)
            logger.info("Initialized assistants for PI and Scientific Critic")
        except Exception as e:
            logger.error(f"Failed to initialize assistants: {e}")
            raise
    
    def conduct_team_meeting(self, agenda: str, hired_agents: Dict[str, BaseAgent], 
                           session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Conduct team meeting using Virtual Lab methodology (Swanson et al. 2023).
        
        Args:
            agenda: Meeting agenda
            hired_agents: Dictionary of hired agents
            session_id: Optional session ID
            
        Returns:
            Meeting results with comprehensive tracking
        """
        if not session_id:
            session_id = f"team_meeting_{int(time.time())}"
        
        logger.info(f"Starting team meeting: {session_id}")
        
        try:
            # Convert hired agents to Virtual Lab agents
            vl_agents = []
            agent_bridges = {}
            
            for agent_id, base_agent in hired_agents.items():
                bridge = VirtualLabAgentBridge(base_agent)
                bridge.initialize_assistant(self.client)
                vl_agents.append(bridge.vl_agent)
                agent_bridges[agent_id] = bridge
            
            # Create save directory
            save_dir = Path("meetings") / session_id
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Run Virtual Lab team meeting
            summary = run_meeting(
                meeting_type="team",
                agenda=agenda,
                save_dir=save_dir,
                save_name="team_meeting",
                team_lead=self.pi_bridge.vl_agent,
                team_members=tuple(vl_agents),
                num_rounds=self.config.get('team_meeting_rounds', 3),
                temperature=CONSISTENT_TEMPERATURE,
                pubmed_search=self.config.get('enable_pubmed_search', True),
                return_summary=True
            )
            
            # Store meeting data
            meeting_data = {
                'session_id': session_id,
                'meeting_type': 'team',
                'agenda': agenda,
                'summary': summary,
                'save_dir': str(save_dir),
                'participants': list(hired_agents.keys()),
                'timestamp': time.time(),
                'success': True
            }
            
            self.meeting_history.append(meeting_data)
            self.active_meetings[session_id] = meeting_data
            
            logger.info(f"Team meeting completed: {session_id}")
            
            return meeting_data
            
        except Exception as e:
            logger.error(f"Team meeting failed: {e}")
            return {
                'session_id': session_id,
                'meeting_type': 'team',
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def conduct_individual_meeting(self, agent: BaseAgent, agenda: str,
                                 session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Conduct individual meeting using Virtual Lab methodology (Swanson et al. 2023).
        
        Args:
            agent: Agent to meet with
            agenda: Meeting agenda
            session_id: Optional session ID
            
        Returns:
            Meeting results
        """
        if not session_id:
            session_id = f"individual_meeting_{int(time.time())}"
        
        logger.info(f"Starting individual meeting with {agent.agent_id}: {session_id}")
        
        try:
            # Create agent bridge
            bridge = VirtualLabAgentBridge(agent)
            bridge.initialize_assistant(self.client)
            
            # Create save directory
            save_dir = Path("meetings") / session_id
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Run Virtual Lab individual meeting
            summary = run_meeting(
                meeting_type="individual",
                agenda=agenda,
                save_dir=save_dir,
                save_name="individual_meeting",
                team_member=bridge.vl_agent,
                num_rounds=self.config.get('individual_meeting_rounds', 2),
                temperature=CONSISTENT_TEMPERATURE,
                return_summary=True
            )
            
            # Store meeting data
            meeting_data = {
                'session_id': session_id,
                'meeting_type': 'individual',
                'agenda': agenda,
                'summary': summary,
                'save_dir': str(save_dir),
                'participant': agent.agent_id,
                'timestamp': time.time(),
                'success': True
            }
            
            self.meeting_history.append(meeting_data)
            self.active_meetings[session_id] = meeting_data
            
            logger.info(f"Individual meeting completed: {session_id}")
            
            return meeting_data
            
        except Exception as e:
            logger.error(f"Individual meeting failed: {e}")
            return {
                'session_id': session_id,
                'meeting_type': 'individual',
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def conduct_research_session(self, research_question: str,
                               constraints: Optional[Dict[str, Any]] = None,
                               context: Optional[Dict[str, Any]] = None,
                               session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Conduct a complete research session using Virtual Lab methodology (Swanson et al. 2023).
        
        Args:
            research_question: Research question to investigate
            constraints: Research constraints
            context: Additional context
            session_id: Optional session ID
            
        Returns:
            Complete research session results
        """
        if not session_id:
            session_id = f"research_session_{int(time.time())}"
        
        self.current_session_id = session_id
        logger.info(f"Starting research session: {session_id}")
        
        try:
            # Hire agents using existing marketplace
            hiring_result = self.agent_marketplace.hire_agents_for_research(
                research_question, constraints or {}
            )
            
            hired_agents = hiring_result.get('hired_agents', {})
            
            if not hired_agents:
                return {
                    'session_id': session_id,
                    'success': False,
                    'error': 'No agents hired for research',
                    'timestamp': time.time()
                }
            
            # Conduct team meeting
            team_agenda = f"Research Question: {research_question}\n\nContext: {context or 'No additional context'}"
            team_meeting_result = self.conduct_team_meeting(
                agenda=team_agenda,
                hired_agents=hired_agents,
                session_id=f"{session_id}_team"
            )
            
            # Conduct individual meetings for detailed analysis
            individual_meetings = {}
            for agent_id, agent in hired_agents.items():
                individual_agenda = f"Detailed analysis of: {research_question}\n\nProvide specific insights from your expertise."
                individual_result = self.conduct_individual_meeting(
                    agent=agent,
                    agenda=individual_agenda,
                    session_id=f"{session_id}_individual_{agent_id}"
                )
                individual_meetings[agent_id] = individual_result
            
            # Compile session results
            session_results = {
                'session_id': session_id,
                'research_question': research_question,
                'constraints': constraints,
                'context': context,
                'team_meeting': team_meeting_result,
                'individual_meetings': individual_meetings,
                'hired_agents': list(hired_agents.keys()),
                'hiring_result': hiring_result,
                'timestamp': time.time(),
                'success': True
            }
            
            # Save session results
            self._save_session_results(session_id, session_results)
            
            logger.info(f"Research session completed: {session_id}")
            
            return session_results
            
        except Exception as e:
            logger.error(f"Research session failed: {e}")
            return {
                'session_id': session_id,
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _save_session_results(self, session_id: str, results: Dict[str, Any]):
        """Save session results to file."""
        try:
            results_dir = Path("sessions") / session_id
            results_dir.mkdir(parents=True, exist_ok=True)
            
            results_file = results_dir / "session_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Saved session results: {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save session results: {e}")
    
    def get_meeting_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get meeting history."""
        if limit:
            return self.meeting_history[-limit:]
        return self.meeting_history
    
    def get_active_meetings(self) -> Dict[str, Any]:
        """Get currently active meetings."""
        return self.active_meetings
    
    def cleanup_session(self, session_id: str):
        """Clean up session resources."""
        if session_id in self.active_meetings:
            del self.active_meetings[session_id]
            logger.info(f"Cleaned up session: {session_id}")
    
    def cleanup(self):
        """Clean up all resources."""
        try:
            self.pi_bridge.cleanup()
            self.critic_bridge.cleanup()
            logger.info("Cleaned up Virtual Lab Meeting System")
        except Exception as e:
            logger.error(f"Failed to cleanup Virtual Lab Meeting System: {e}")
