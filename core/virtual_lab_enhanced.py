"""
Enhanced Virtual Lab - AI-Human Research Collaboration System

This module integrates the proven Virtual Lab methodology from the Nature paper
with the existing AI research lab framework to create a robust, scientifically
rigorous meeting-based research coordination system.

Key enhancements:
- Virtual Lab's proven meeting methodology
- OpenAI Assistants API integration
- PubMed search tool integration
- Enhanced agent coordination
- Scientific critique integration
- Comprehensive cost tracking
- Discussion persistence (JSON/Markdown)
"""

import logging
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Literal
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI
from tqdm import trange, tqdm

from agents import PrincipalInvestigatorAgent, ScientificCriticAgent, AgentMarketplace
from agents.base_agent import BaseAgent
from data.literature_retriever import LiteratureRetriever

# Import Virtual Lab components
from core.virtual_lab_integration.agent import Agent as VirtualLabAgent
from core.virtual_lab_integration.constants import CONSISTENT_TEMPERATURE, PUBMED_TOOL_DESCRIPTION
from core.virtual_lab_integration.prompts import (
    individual_meeting_agent_prompt,
    individual_meeting_critic_prompt,
    individual_meeting_start_prompt,
    SCIENTIFIC_CRITIC as VL_SCIENTIFIC_CRITIC,
    team_meeting_start_prompt,
    team_meeting_team_lead_initial_prompt,
    team_meeting_team_lead_intermediate_prompt,
    team_meeting_team_lead_final_prompt,
    team_meeting_team_member_prompt,
    PRINCIPAL_INVESTIGATOR as VL_PRINCIPAL_INVESTIGATOR,
)
from core.virtual_lab_integration.utils import (
    convert_messages_to_discussion,
    count_discussion_tokens,
    count_tokens,
    get_messages,
    get_summary,
    print_cost_and_time,
    run_tools,
    save_meeting,
)

logger = logging.getLogger(__name__)


class MeetingType(Enum):
    """Types of meetings in the Enhanced Virtual Lab."""
    TEAM_MEETING = "team_meeting"
    INDIVIDUAL_MEETING = "individual_meeting"
    AGGREGATION_MEETING = "aggregation_meeting"


class ResearchPhase(Enum):
    """Structured research phases inspired by the Virtual Lab paper."""
    TEAM_SELECTION = "team_selection"
    LITERATURE_REVIEW = "literature_review"
    PROJECT_SPECIFICATION = "project_specification"  
    TOOLS_SELECTION = "tools_selection"
    TOOLS_IMPLEMENTATION = "tools_implementation"
    WORKFLOW_DESIGN = "workflow_design"
    EXECUTION = "execution"
    SYNTHESIS = "synthesis"


@dataclass
class MeetingAgenda:
    """Enhanced structure for meeting agendas with Virtual Lab integration."""
    meeting_id: str
    meeting_type: MeetingType
    phase: ResearchPhase
    objectives: List[str]
    participants: List[str]
    discussion_topics: List[str]
    expected_outcomes: List[str]
    duration_minutes: int = 10
    agenda_questions: Tuple[str, ...] = ()
    agenda_rules: Tuple[str, ...] = ()
    summaries: Tuple[str, ...] = ()
    contexts: Tuple[str, ...] = ()
    num_rounds: int = 1
    temperature: float = CONSISTENT_TEMPERATURE
    pubmed_search: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'meeting_id': self.meeting_id,
            'meeting_type': self.meeting_type.value,
            'phase': self.phase.value,
            'objectives': self.objectives,
            'participants': self.participants,
            'discussion_topics': self.discussion_topics,
            'expected_outcomes': self.expected_outcomes,
            'duration_minutes': self.duration_minutes,
            'agenda_questions': self.agenda_questions,
            'agenda_rules': self.agenda_rules,
            'summaries': self.summaries,
            'contexts': self.contexts,
            'num_rounds': self.num_rounds,
            'temperature': self.temperature,
            'pubmed_search': self.pubmed_search
        }


@dataclass
class MeetingRecord:
    """Enhanced record of a completed meeting with Virtual Lab integration."""
    meeting_id: str
    meeting_type: MeetingType
    phase: ResearchPhase
    participants: List[str]
    agenda: MeetingAgenda
    discussion_transcript: List[Dict[str, Any]]
    outcomes: Dict[str, Any]
    decisions: List[str]
    action_items: List[str]
    start_time: float
    end_time: float
    success: bool
    token_counts: Dict[str, int]
    cost: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'meeting_id': self.meeting_id,
            'meeting_type': self.meeting_type.value,
            'phase': self.phase.value,
            'participants': self.participants,
            'agenda': self.agenda.to_dict(),
            'discussion_transcript': self.discussion_transcript,
            'outcomes': self.outcomes,
            'decisions': self.decisions,
            'action_items': self.action_items,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'success': self.success,
            'token_counts': self.token_counts,
            'cost': self.cost
        }


class EnhancedVirtualLabMeetingSystem:
    """
    Enhanced Virtual Lab Meeting System
    
    Integrates the proven Virtual Lab methodology with the existing AI research lab framework.
    Provides scientifically rigorous meeting-based research coordination with comprehensive
    cost tracking, tool integration, and discussion persistence.
    """
    
    def __init__(self, pi_agent: PrincipalInvestigatorAgent, 
                 scientific_critic: ScientificCriticAgent,
                 agent_marketplace: AgentMarketplace,
                 config: Optional[Dict[str, Any]] = None,
                 cost_manager=None):
        """
        Initialize the Enhanced Virtual Lab Meeting System.
        
        Args:
            pi_agent: Principal Investigator agent
            scientific_critic: Scientific Critic agent
            agent_marketplace: Agent marketplace for hiring
            config: Configuration dictionary
            cost_manager: Cost management system
        """
        self.pi_agent = pi_agent
        self.scientific_critic = scientific_critic
        self.agent_marketplace = agent_marketplace
        self.config = config or {}
        self.cost_manager = cost_manager
        
        # Initialize OpenAI client
        self.client = OpenAI()
        
        # Session management
        self.current_session_id = None
        self.meeting_history = []
        self.agent_performance = {}
        
        # Virtual Lab integration
        self.vl_agents = {}
        self.assistant_cache = {}
        
        # Logging setup
        self.setup_logging()
        
        logger.info("Enhanced Virtual Lab Meeting System initialized")
    
    def setup_logging(self):
        """Setup logging for the enhanced system."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "enhanced_virtual_lab.log"),
                logging.StreamHandler()
            ]
        )
    
    def create_virtual_lab_agent(self, base_agent: BaseAgent) -> VirtualLabAgent:
        """
        Convert a BaseAgent to a Virtual Lab Agent.
        
        Args:
            base_agent: Base agent to convert
            
        Returns:
            Virtual Lab Agent
        """
        # Extract expertise from base agent
        expertise = ", ".join(base_agent.expertise) if hasattr(base_agent, 'expertise') else "general expertise"
        
        # Create Virtual Lab agent
        vl_agent = VirtualLabAgent(
            title=base_agent.agent_id,
            expertise=expertise,
            goal="contribute specialized knowledge to research discussions",
            role="provide expert input and analysis",
            model="gpt-4o-2024-08-06"  # Default to latest model
        )
        
        return vl_agent
    
    def get_or_create_assistant(self, agent: VirtualLabAgent, pubmed_search: bool = False):
        """
        Get or create an OpenAI assistant for the agent.
        
        Args:
            agent: Virtual Lab agent
            pubmed_search: Whether to include PubMed search tool
            
        Returns:
            OpenAI assistant
        """
        cache_key = f"{agent.title}_{pubmed_search}"
        
        if cache_key in self.assistant_cache:
            return self.assistant_cache[cache_key]
        
        # Set up tools
        assistant_params = {"tools": [PUBMED_TOOL_DESCRIPTION]} if pubmed_search else {}
        
        # Create assistant
        assistant = self.client.beta.assistants.create(
            name=agent.title,
            instructions=agent.prompt,
            model=agent.model,
            **assistant_params,
        )
        
        self.assistant_cache[cache_key] = assistant
        return assistant
    
    def run_enhanced_meeting(self, agenda: MeetingAgenda, 
                           hired_agents: Dict[str, BaseAgent],
                           research_question: str, 
                           constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run an enhanced meeting using Virtual Lab methodology.
        
        Args:
            agenda: Meeting agenda
            hired_agents: Dictionary of hired agents
            research_question: Research question
            constraints: Research constraints
            
        Returns:
            Meeting results with comprehensive tracking
        """
        meeting_id = agenda.meeting_id
        session_id = self.current_session_id
        
        logger.info(f"Starting enhanced meeting: {meeting_id}")
        
        # Convert base agents to Virtual Lab agents
        vl_agents = {}
        for agent_id, base_agent in hired_agents.items():
            vl_agents[agent_id] = self.create_virtual_lab_agent(base_agent)
        
        # Add PI and Scientific Critic
        vl_agents['pi'] = VL_PRINCIPAL_INVESTIGATOR
        vl_agents['critic'] = VL_SCIENTIFIC_CRITIC
        
        # Determine meeting type
        meeting_type = "team" if agenda.meeting_type == MeetingType.TEAM_MEETING else "individual"
        
        # Set up team for Virtual Lab meeting
        if meeting_type == "team":
            team_lead = vl_agents['pi']
            team_members = tuple(vl_agents.values())
        else:
            team_lead = None
            team_members = None
            team_member = vl_agents.get(list(hired_agents.keys())[0], vl_agents['pi'])
        
        # Create save directory
        save_dir = Path("meetings") / session_id if session_id else Path("meetings")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Run Virtual Lab meeting
        try:
            if meeting_type == "team":
                summary = self._run_virtual_lab_team_meeting(
                    agenda=agenda.agenda_questions[0] if agenda.agenda_questions else research_question,
                    save_dir=save_dir,
                    save_name=meeting_id,
                    team_lead=team_lead,
                    team_members=tuple(vl_agents.values()),
                    agenda_questions=agenda.agenda_questions,
                    agenda_rules=agenda.agenda_rules,
                    summaries=agenda.summaries,
                    contexts=agenda.contexts,
                    num_rounds=agenda.num_rounds,
                    temperature=agenda.temperature,
                    pubmed_search=agenda.pubmed_search,
                    return_summary=True
                )
            else:
                summary = self._run_virtual_lab_individual_meeting(
                    agenda=agenda.agenda_questions[0] if agenda.agenda_questions else research_question,
                    save_dir=save_dir,
                    save_name=meeting_id,
                    team_member=team_member,
                    agenda_questions=agenda.agenda_questions,
                    agenda_rules=agenda.agenda_rules,
                    summaries=agenda.summaries,
                    contexts=agenda.contexts,
                    temperature=agenda.temperature,
                    pubmed_search=agenda.pubmed_search,
                    return_summary=True
                )
            
            # Create meeting record
            meeting_record = MeetingRecord(
                meeting_id=meeting_id,
                meeting_type=agenda.meeting_type,
                phase=agenda.phase,
                participants=list(vl_agents.keys()),
                agenda=agenda,
                discussion_transcript=[],  # Will be loaded from saved discussion
                outcomes={'summary': summary},
                decisions=[],
                action_items=[],
                start_time=time.time(),
                end_time=time.time(),
                success=True,
                token_counts={},
                cost=0.0
            )
            
            self.meeting_history.append(meeting_record)
            
            return {
                'success': True,
                'meeting_id': meeting_id,
                'summary': summary,
                'meeting_record': meeting_record.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Enhanced meeting failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'meeting_id': meeting_id
            }
    
    def _run_virtual_lab_team_meeting(self, agenda: str, save_dir: Path, save_name: str,
                                     team_lead: VirtualLabAgent, team_members: Tuple[VirtualLabAgent, ...],
                                     agenda_questions: Tuple[str, ...] = (), agenda_rules: Tuple[str, ...] = (),
                                     summaries: Tuple[str, ...] = (), contexts: Tuple[str, ...] = (),
                                     num_rounds: int = 1, temperature: float = CONSISTENT_TEMPERATURE,
                                     pubmed_search: bool = False, return_summary: bool = False) -> str:
        """
        Run a Virtual Lab team meeting.
        
        Args:
            agenda: Meeting agenda
            save_dir: Directory to save discussion
            save_name: Name for saved discussion
            team_lead: Team lead agent
            team_members: Team member agents
            agenda_questions: Questions to answer
            agenda_rules: Meeting rules
            summaries: Previous meeting summaries
            contexts: Meeting contexts
            num_rounds: Number of discussion rounds
            temperature: Sampling temperature
            pubmed_search: Whether to include PubMed search
            return_summary: Whether to return summary
            
        Returns:
            Meeting summary if return_summary is True
        """
        # Validate meeting parameters
        if team_lead is None or team_members is None or len(team_members) == 0:
            raise ValueError("Team meeting requires team lead and team members")
        if team_lead in team_members:
            raise ValueError("Team lead must be separate from team members")
        if len(set(team_members)) != len(team_members):
            raise ValueError("Team members must be unique")
        
        # Start timing
        start_time = time.time()
        
        # Set up team
        team = [team_lead] + list(team_members)
        
        # Set up tools
        assistant_params = {"tools": [PUBMED_TOOL_DESCRIPTION]} if pubmed_search else {}
        
        # Set up the assistants
        agent_to_assistant = {
            agent: self.get_or_create_assistant(agent, pubmed_search)
            for agent in team
        }
        
        # Map assistant IDs to agents
        assistant_id_to_title = {
            assistant.id: agent.title for agent, assistant in agent_to_assistant.items()
        }
        
        # Set up tool token count
        tool_token_count = 0
        
        # Set up the thread
        thread = self.client.beta.threads.create()
        
        # Initial prompt for team meeting
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=team_meeting_start_prompt(
                team_lead=team_lead,
                team_members=team_members,
                agenda=agenda,
                agenda_questions=agenda_questions,
                agenda_rules=agenda_rules,
                summaries=summaries,
                contexts=contexts,
                num_rounds=num_rounds,
            ),
        )
        
        # Loop through rounds
        for round_index in trange(num_rounds + 1, desc="Rounds (+ Final Round)"):
            round_num = round_index + 1
            
            # Loop through team and elicit responses
            for agent in tqdm(team, desc="Team"):
                # Prompt based on agent and round number
                if agent == team_lead:
                    if round_index == 0:
                        prompt = team_meeting_team_lead_initial_prompt(team_lead=team_lead)
                    elif round_index == num_rounds:
                        prompt = team_meeting_team_lead_final_prompt(
                            team_lead=team_lead,
                            agenda=agenda,
                            agenda_questions=agenda_questions,
                            agenda_rules=agenda_rules,
                        )
                    else:
                        prompt = team_meeting_team_lead_intermediate_prompt(
                            team_lead=team_lead,
                            round_num=round_num - 1,
                            num_rounds=num_rounds,
                        )
                else:
                    prompt = team_meeting_team_member_prompt(
                        team_member=agent, round_num=round_num, num_rounds=num_rounds
                    )
                
                # Create message from user to agent
                self.client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=prompt,
                )
                
                # Run the agent
                run = self.client.beta.threads.runs.create_and_poll(
                    thread_id=thread.id,
                    assistant_id=agent_to_assistant[agent].id,
                    model=agent.model,
                    temperature=temperature,
                )
                
                # Check if run requires action
                if run.status == "requires_action":
                    # Run the tools
                    tool_outputs = run_tools(run=run)
                    
                    # Update tool token count
                    tool_token_count += sum(
                        count_tokens(tool_output["output"]) for tool_output in tool_outputs
                    )
                    
                    # Submit the tool outputs
                    run = self.client.beta.threads.runs.submit_tool_outputs_and_poll(
                        thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs
                    )
                    
                    # Add tool outputs to the thread
                    self.client.beta.threads.messages.create(
                        thread_id=thread.id,
                        role="user",
                        content="Tool Output:\n\n"
                        + "\n\n".join(
                            tool_output["output"] for tool_output in tool_outputs
                        ),
                    )
                
                # Check run status
                if run.status != "completed":
                    raise ValueError(f"Run failed: {run.status}")
                
                # If final round, only team lead responds
                if round_index == num_rounds:
                    break
        
        # Get messages from the discussion
        messages = get_messages(client=self.client, thread_id=thread.id)
        
        # Convert messages to discussion format
        discussion = convert_messages_to_discussion(
            messages=messages, assistant_id_to_title=assistant_id_to_title
        )
        
        # Count discussion tokens
        token_counts = count_discussion_tokens(discussion=discussion)
        
        # Add tool token count to total token count
        token_counts["tool"] = tool_token_count
        
        # Print cost and time
        print_cost_and_time(
            token_counts=token_counts,
            model=team_lead.model,
            elapsed_time=time.time() - start_time,
        )
        
        # Save the discussion as JSON and Markdown
        save_meeting(
            save_dir=save_dir,
            save_name=save_name,
            discussion=discussion,
        )
        
        # Optionally, return summary
        if return_summary:
            return get_summary(discussion)
        return None
    
    def _run_virtual_lab_individual_meeting(self, agenda: str, save_dir: Path, save_name: str,
                                          team_member: VirtualLabAgent,
                                          agenda_questions: Tuple[str, ...] = (), agenda_rules: Tuple[str, ...] = (),
                                          summaries: Tuple[str, ...] = (), contexts: Tuple[str, ...] = (),
                                          temperature: float = CONSISTENT_TEMPERATURE,
                                          pubmed_search: bool = False, return_summary: bool = False) -> str:
        """
        Run a Virtual Lab individual meeting.
        
        Args:
            agenda: Meeting agenda
            save_dir: Directory to save discussion
            save_name: Name for saved discussion
            team_member: Team member agent
            agenda_questions: Questions to answer
            agenda_rules: Meeting rules
            summaries: Previous meeting summaries
            contexts: Meeting contexts
            temperature: Sampling temperature
            pubmed_search: Whether to include PubMed search
            return_summary: Whether to return summary
            
        Returns:
            Meeting summary if return_summary is True
        """
        # Validate meeting parameters
        if team_member is None:
            raise ValueError("Individual meeting requires individual team member")
        
        # Start timing
        start_time = time.time()
        
        # Set up team
        team = [team_member] + [VL_SCIENTIFIC_CRITIC]
        
        # Set up tools
        assistant_params = {"tools": [PUBMED_TOOL_DESCRIPTION]} if pubmed_search else {}
        
        # Set up the assistants
        agent_to_assistant = {
            agent: self.get_or_create_assistant(agent, pubmed_search)
            for agent in team
        }
        
        # Map assistant IDs to agents
        assistant_id_to_title = {
            assistant.id: agent.title for agent, assistant in agent_to_assistant.items()
        }
        
        # Set up tool token count
        tool_token_count = 0
        
        # Set up the thread
        thread = self.client.beta.threads.create()
        
        # Initial prompt for individual meeting
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=individual_meeting_start_prompt(
                team_member=team_member,
                agenda=agenda,
                agenda_questions=agenda_questions,
                agenda_rules=agenda_rules,
                summaries=summaries,
                contexts=contexts,
            ),
        )
        
        # Run the meeting (individual meetings are simpler)
        for agent in team:
            # Prompt based on agent
            if agent == VL_SCIENTIFIC_CRITIC:
                prompt = individual_meeting_critic_prompt(
                    critic=VL_SCIENTIFIC_CRITIC, agent=team_member
                )
            else:
                prompt = individual_meeting_agent_prompt(
                    critic=VL_SCIENTIFIC_CRITIC, agent=team_member
                )
            
            # Create message from user to agent
            self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt,
            )
            
            # Run the agent
            run = self.client.beta.threads.runs.create_and_poll(
                thread_id=thread.id,
                assistant_id=agent_to_assistant[agent].id,
                model=agent.model,
                temperature=temperature,
            )
            
            # Check if run requires action
            if run.status == "requires_action":
                # Run the tools
                tool_outputs = run_tools(run=run)
                
                # Update tool token count
                tool_token_count += sum(
                    count_tokens(tool_output["output"]) for tool_output in tool_outputs
                )
                
                # Submit the tool outputs
                run = self.client.beta.threads.runs.submit_tool_outputs_and_poll(
                    thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs
                )
                
                # Add tool outputs to the thread
                self.client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content="Tool Output:\n\n"
                    + "\n\n".join(
                        tool_output["output"] for tool_output in tool_outputs
                    ),
                )
            
            # Check run status
            if run.status != "completed":
                raise ValueError(f"Run failed: {run.status}")
        
        # Get messages from the discussion
        messages = get_messages(client=self.client, thread_id=thread.id)
        
        # Convert messages to discussion format
        discussion = convert_messages_to_discussion(
            messages=messages, assistant_id_to_title=assistant_id_to_title
        )
        
        # Count discussion tokens
        token_counts = count_discussion_tokens(discussion=discussion)
        
        # Add tool token count to total token count
        token_counts["tool"] = tool_token_count
        
        # Print cost and time
        print_cost_and_time(
            token_counts=token_counts,
            model=team_member.model,
            elapsed_time=time.time() - start_time,
        )
        
        # Save the discussion as JSON and Markdown
        save_meeting(
            save_dir=save_dir,
            save_name=save_name,
            discussion=discussion,
        )
        
        # Optionally, return summary
        if return_summary:
            return get_summary(discussion)
        return None
    
    def conduct_research_session(self, research_question: str,
                               constraints: Optional[Dict[str, Any]] = None,
                               context: Optional[Dict[str, Any]] = None,
                               session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Conduct a research session using enhanced Virtual Lab methodology.
        
        Args:
            research_question: Research question to investigate
            constraints: Research constraints
            context: Additional context
            session_id: Session identifier
            
        Returns:
            Research session results
        """
        # Set session ID
        self.current_session_id = session_id or f"session_{int(time.time())}"
        
        logger.info(f"Starting enhanced research session: {self.current_session_id}")
        
        # Initialize session data
        session_data = {
            'session_id': self.current_session_id,
            'research_question': research_question,
            'constraints': constraints or {},
            'context': context or {},
            'phases': {},
            'meetings': [],
            'final_results': {}
        }
        
        # Execute research phases with enhanced meeting system
        for phase in ResearchPhase:
            logger.info(f"Executing phase: {phase.value}")
            
            phase_result = self._execute_enhanced_research_phase(
                phase=phase,
                session_id=self.current_session_id,
                research_question=research_question,
                constraints=constraints or {}
            )
            
            session_data['phases'][phase.value] = phase_result
            
            # Add meetings from this phase
            if 'meetings' in phase_result:
                session_data['meetings'].extend(phase_result['meetings'])
        
        # Compile final results
        session_data['final_results'] = self._compile_enhanced_final_results(session_data)
        
        logger.info(f"Enhanced research session completed: {self.current_session_id}")
        
        return session_data
    
    def _execute_enhanced_research_phase(self, phase: ResearchPhase, session_id: str,
                                       research_question: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an enhanced research phase using Virtual Lab methodology.
        
        Args:
            phase: Research phase to execute
            session_id: Session identifier
            research_question: Research question
            constraints: Research constraints
            
        Returns:
            Phase execution results
        """
        logger.info(f"Executing enhanced phase: {phase.value}")
        
        # Create enhanced meeting agenda
        agenda = MeetingAgenda(
            meeting_id=f"{phase.value}_{session_id}",
            meeting_type=MeetingType.TEAM_MEETING,
            phase=phase,
            objectives=[f"Execute {phase.value} phase for research question"],
            participants=[],
            discussion_topics=[research_question],
            expected_outcomes=[f"Complete {phase.value} phase"],
            agenda_questions=(research_question,),
            agenda_rules=("Maintain scientific rigor", "Follow research methodology"),
            num_rounds=2,
            temperature=CONSISTENT_TEMPERATURE,
            pubmed_search=True
        )
        
        # Hire agents for this phase
        hired_agents = self._hire_agents_for_phase(phase, constraints)
        
        # Run enhanced meeting
        meeting_result = self.run_enhanced_meeting(
            agenda=agenda,
            hired_agents=hired_agents,
            research_question=research_question,
            constraints=constraints
        )
        
        return {
            'phase': phase.value,
            'meeting_result': meeting_result,
            'hired_agents': {k: v.agent_id for k, v in hired_agents.items()},
            'success': meeting_result.get('success', False)
        }
    
    def _hire_agents_for_phase(self, phase: ResearchPhase, constraints: Dict[str, Any]) -> Dict[str, BaseAgent]:
        """
        Hire agents for a specific research phase.
        
        Args:
            phase: Research phase
            constraints: Research constraints
            
        Returns:
            Dictionary of hired agents
        """
        # Define expertise requirements for each phase
        phase_expertise = {
            ResearchPhase.TEAM_SELECTION: ["team_management", "project_planning"],
            ResearchPhase.LITERATURE_REVIEW: ["literature_analysis", "research_methodology"],
            ResearchPhase.PROJECT_SPECIFICATION: ["project_management", "requirements_analysis"],
            ResearchPhase.TOOLS_SELECTION: ["software_engineering", "tool_evaluation"],
            ResearchPhase.TOOLS_IMPLEMENTATION: ["software_development", "system_integration"],
            ResearchPhase.WORKFLOW_DESIGN: ["process_design", "workflow_optimization"],
            ResearchPhase.EXECUTION: ["execution_management", "quality_assurance"],
            ResearchPhase.SYNTHESIS: ["data_analysis", "synthesis", "reporting"]
        }
        
        required_expertise = phase_expertise.get(phase, ["general_expertise"])
        
        # Hire agents from marketplace
        hired_agents = {}
        for expertise in required_expertise:
            agent = self.agent_marketplace.hire_agent(expertise, constraints)
            if agent:
                hired_agents[f"{expertise}_agent"] = agent
        
        return hired_agents
    
    def _compile_enhanced_final_results(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compile final results from enhanced research session.
        
        Args:
            session_data: Session data
            
        Returns:
            Compiled final results
        """
        # Extract summaries from all meetings
        summaries = []
        for meeting in session_data.get('meetings', []):
            if 'summary' in meeting:
                summaries.append(meeting['summary'])
        
        # Compile phase results
        phase_results = {}
        for phase_name, phase_data in session_data.get('phases', {}).items():
            if phase_data.get('success'):
                phase_results[phase_name] = {
                    'status': 'completed',
                    'summary': phase_data.get('meeting_result', {}).get('summary', '')
                }
            else:
                phase_results[phase_name] = {
                    'status': 'failed',
                    'error': phase_data.get('meeting_result', {}).get('error', 'Unknown error')
                }
        
        return {
            'research_question': session_data.get('research_question'),
            'session_id': session_data.get('session_id'),
            'phase_results': phase_results,
            'meeting_summaries': summaries,
            'total_meetings': len(session_data.get('meetings', [])),
            'success_rate': len([p for p in phase_results.values() if p['status'] == 'completed']) / len(phase_results) if phase_results else 0
        }
    
    def get_meeting_history(self, limit: Optional[int] = None) -> List[MeetingRecord]:
        """
        Get meeting history.
        
        Args:
            limit: Maximum number of meetings to return
            
        Returns:
            List of meeting records
        """
        if limit:
            return self.meeting_history[-limit:]
        return self.meeting_history.copy()
    
    def get_meeting_statistics(self) -> Dict[str, Any]:
        """
        Get meeting statistics.
        
        Returns:
            Meeting statistics
        """
        if not self.meeting_history:
            return {}
        
        total_meetings = len(self.meeting_history)
        successful_meetings = len([m for m in self.meeting_history if m.success])
        total_cost = sum(m.cost for m in self.meeting_history)
        avg_cost = total_cost / total_meetings if total_meetings > 0 else 0
        
        return {
            'total_meetings': total_meetings,
            'successful_meetings': successful_meetings,
            'success_rate': successful_meetings / total_meetings if total_meetings > 0 else 0,
            'total_cost': total_cost,
            'average_cost': avg_cost,
            'meeting_types': {
                meeting_type.value: len([m for m in self.meeting_history if m.meeting_type == meeting_type])
                for meeting_type in MeetingType
            }
        } 