"""
Pure Virtual Lab Meeting System

A meeting system that works exclusively with Virtual Lab agents and methodology.
This eliminates the bridge pattern and uses only Virtual Lab agents for all interactions.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

from openai import OpenAI

from core.virtual_lab_integration.agent import Agent
from core.virtual_lab_integration.run_meeting import run_meeting
from core.virtual_lab_integration.constants import CONSISTENT_TEMPERATURE
from agents.virtual_lab_agent_marketplace import VirtualLabAgentMarketplace

logger = logging.getLogger(__name__)


class VirtualLabMeetingSystemVL:
    """
    Pure Virtual Lab meeting system.
    
    This system works exclusively with Virtual Lab agents and methodology,
    eliminating the bridge pattern and dual agent system.
    """
    
    def __init__(self, pi_agent: Agent, scientific_critic: Agent, 
                 agent_marketplace: VirtualLabAgentMarketplace, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pure Virtual Lab meeting system.
        
        Args:
            pi_agent: Principal Investigator Virtual Lab agent
            scientific_critic: Scientific Critic Virtual Lab agent
            agent_marketplace: Virtual Lab agent marketplace
            config: Configuration dictionary
        """
        self.pi_agent = pi_agent
        self.scientific_critic = scientific_critic
        self.agent_marketplace = agent_marketplace
        self.config = config or {}
        
        # Initialize OpenAI client
        self.client = OpenAI()
        
        # Session tracking
        self.current_session_id = None
        self.meeting_history = []
        self.active_meetings = {}
        
        logger.info("Pure Virtual Lab Meeting System initialized")
    
    def conduct_team_meeting(self, agenda: str, hired_agents: Dict[str, Agent], 
                           session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Conduct team meeting using Virtual Lab agents only.
        
        Args:
            agenda: Meeting agenda
            hired_agents: Dictionary of hired Virtual Lab agents
            session_id: Optional session ID
            
        Returns:
            Meeting results with comprehensive tracking
        """
        if not session_id:
            session_id = f"team_meeting_{int(time.time())}"
        
        logger.info(f"Starting team meeting: {session_id}")
        
        try:
            # Convert hired agents to list for Virtual Lab meeting
            vl_agents = list(hired_agents.values())
            
            # Create save directory
            save_dir = Path("meetings") / session_id
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Run Virtual Lab team meeting
            summary = run_meeting(
                meeting_type="team",
                agenda=agenda,
                save_dir=save_dir,
                save_name="team_meeting",
                team_lead=self.pi_agent,
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
    
    def conduct_individual_meeting(self, agent: Agent, agenda: str,
                                 session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Conduct individual meeting using Virtual Lab agent.
        
        Args:
            agent: Virtual Lab agent to meet with
            agenda: Meeting agenda
            session_id: Optional session ID
            
        Returns:
            Meeting results
        """
        if not session_id:
            session_id = f"individual_meeting_{int(time.time())}"
        
        logger.info(f"Starting individual meeting with {agent.title}: {session_id}")
        
        try:
            # Create save directory
            save_dir = Path("meetings") / session_id
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Run Virtual Lab individual meeting
            summary = run_meeting(
                meeting_type="individual",
                agenda=agenda,
                save_dir=save_dir,
                save_name="individual_meeting",
                team_member=agent,
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
                'participant': agent.title,
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
        Conduct a complete research session using Virtual Lab agents only.
        
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
            # Hire Virtual Lab agents using marketplace
            hiring_result = self.agent_marketplace.hire_agents_for_research(
                research_question, constraints or {}
            )
            
            if not hiring_result['success']:
                return {
                    'session_id': session_id,
                    'success': False,
                    'error': hiring_result.get('error', 'Failed to hire agents'),
                    'timestamp': time.time()
                }
            
            hired_agents = hiring_result['hired_agents']
            
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
            for agent_title, agent in hired_agents.items():
                individual_agenda = f"Detailed analysis of: {research_question}\n\nProvide specific insights from your expertise."
                individual_result = self.conduct_individual_meeting(
                    agent=agent,
                    agenda=individual_agenda,
                    session_id=f"{session_id}_individual_{agent_title}"
                )
                individual_meetings[agent_title] = individual_result
            
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
    
    def conduct_critical_review(self, research_output: str, 
                              session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Conduct critical review using Scientific Critic agent.
        
        Args:
            research_output: Research output to review
            session_id: Optional session ID
            
        Returns:
            Critical review results
        """
        if not session_id:
            session_id = f"critical_review_{int(time.time())}"
        
        logger.info(f"Starting critical review: {session_id}")
        
        try:
            # Create review agenda
            review_agenda = f"""
            Critical Review of Research Output
            
            Research Output:
            {research_output}
            
            Please provide a comprehensive critical review including:
            1. Scientific rigor assessment
            2. Methodology evaluation
            3. Evidence quality analysis
            4. Logical consistency check
            5. Recommendations for improvement
            """
            
            # Conduct individual meeting with Scientific Critic
            review_result = self.conduct_individual_meeting(
                agent=self.scientific_critic,
                agenda=review_agenda,
                session_id=session_id
            )
            
            return review_result
            
        except Exception as e:
            logger.error(f"Critical review failed: {e}")
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
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get summary of a specific session.
        
        Args:
            session_id: Session ID to summarize
            
        Returns:
            Session summary or None if not found
        """
        # Look in active meetings first
        if session_id in self.active_meetings:
            return self.active_meetings[session_id]
        
        # Look in meeting history
        for meeting in self.meeting_history:
            if meeting.get('session_id') == session_id:
                return meeting
        
        return None
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions with basic information.
        
        Returns:
            List of session information
        """
        sessions = []
        
        # Add active sessions
        for session_id, meeting_data in self.active_meetings.items():
            sessions.append({
                'session_id': session_id,
                'meeting_type': meeting_data.get('meeting_type'),
                'timestamp': meeting_data.get('timestamp'),
                'status': 'active',
                'success': meeting_data.get('success', False)
            })
        
        # Add historical sessions
        for meeting_data in self.meeting_history:
            session_id = meeting_data.get('session_id')
            if session_id not in self.active_meetings:
                sessions.append({
                    'session_id': session_id,
                    'meeting_type': meeting_data.get('meeting_type'),
                    'timestamp': meeting_data.get('timestamp'),
                    'status': 'completed',
                    'success': meeting_data.get('success', False)
                })
        
        # Sort by timestamp (newest first)
        sessions.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        return sessions
