"""
Virtual Lab - AI-Human Research Collaboration System

Inspired by the paper "The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies"
by Swanson et al., this module implements a meeting-based research coordination system
where AI agents collaborate through structured meetings to conduct sophisticated,
interdisciplinary research.

Key features:
- Meeting-based interactions (team meetings and individual meetings)
- Structured research phases
- Cross-agent interaction and critique
- Iterative refinement workflows
- Scientific critique integration
- Minimal human input requirement
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from agents import PrincipalInvestigatorAgent, ScientificCriticAgent, AgentMarketplace
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class MeetingType(Enum):
    """Types of meetings in the Virtual Lab."""
    TEAM_MEETING = "team_meeting"
    INDIVIDUAL_MEETING = "individual_meeting"
    AGGREGATION_MEETING = "aggregation_meeting"


class ResearchPhase(Enum):
    """Structured research phases inspired by the Virtual Lab paper."""
    TEAM_SELECTION = "team_selection"
    PROJECT_SPECIFICATION = "project_specification"  
    TOOLS_SELECTION = "tools_selection"
    TOOLS_IMPLEMENTATION = "tools_implementation"
    WORKFLOW_DESIGN = "workflow_design"
    EXECUTION = "execution"
    SYNTHESIS = "synthesis"


@dataclass
class MeetingAgenda:
    """Structure for meeting agendas."""
    meeting_id: str
    meeting_type: MeetingType
    phase: ResearchPhase
    objectives: List[str]
    participants: List[str]
    discussion_topics: List[str]
    expected_outcomes: List[str]
    duration_minutes: int = 10


@dataclass
class MeetingRecord:
    """Record of a completed meeting."""
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


class VirtualLabMeetingSystem:
    """
    Meeting-based coordination system for AI research collaboration.
    
    Implements the Virtual Lab approach where research is conducted through
    structured meetings between AI agents with different expertise.
    """
    
    def __init__(self, pi_agent: PrincipalInvestigatorAgent, 
                 scientific_critic: ScientificCriticAgent,
                 agent_marketplace: AgentMarketplace,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Virtual Lab meeting system.
        
        Args:
            pi_agent: Principal Investigator agent for coordination
            scientific_critic: Scientific Critic agent for quality control
            agent_marketplace: Marketplace for hiring domain experts
            config: Optional configuration
        """
        self.pi_agent = pi_agent
        self.scientific_critic = scientific_critic
        self.agent_marketplace = agent_marketplace
        self.config = config or {}
        
        # Meeting management
        self.active_meetings = {}
        self.meeting_history = []
        self.research_sessions = {}
        
        # Phase management
        self.current_phase = None
        self.phase_results = {}
        
        logger.info("Virtual Lab Meeting System initialized")
    
    def conduct_research_session(self, research_question: str,
                                constraints: Optional[Dict[str, Any]] = None,
                                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Conduct a complete Virtual Lab research session using structured meetings.
        
        Args:
            research_question: The research question to investigate
            constraints: Optional constraints (budget, time, etc.)
            context: Optional additional context
            
        Returns:
            Complete research session results
        """
        session_id = f"vlab_session_{int(time.time())}"
        logger.info(f"Starting Virtual Lab research session: {session_id}")
        
        session_data = {
            'session_id': session_id,
            'research_question': research_question,
            'constraints': constraints or {},
            'context': context or {},
            'start_time': time.time(),
            'status': 'running',
            'phases': {},
            'meetings': [],
            'final_results': {}
        }
        
        try:
            # Execute structured research phases
            for phase in ResearchPhase:
                logger.info(f"Starting phase: {phase.value}")
                self.current_phase = phase
                
                phase_result = self._execute_research_phase(
                    phase, session_id, research_question, constraints
                )
                
                session_data['phases'][phase.value] = phase_result
                self.phase_results[phase.value] = phase_result
                
                if not phase_result.get('success', False):
                    logger.error(f"Phase {phase.value} failed")
                    session_data['status'] = 'failed'
                    break
            
            # Compile final results if all phases succeeded
            if session_data['status'] == 'running':
                session_data['final_results'] = self._compile_final_results(session_data)
                session_data['status'] = 'completed'
            
        except Exception as e:
            logger.error(f"Research session {session_id} failed: {e}")
            session_data['status'] = 'failed'
            session_data['error'] = str(e)
        
        session_data['end_time'] = time.time()
        session_data['duration'] = session_data['end_time'] - session_data['start_time']
        
        # Store session
        self.research_sessions[session_id] = session_data
        
        return session_data
    
    def _execute_research_phase(self, phase: ResearchPhase, session_id: str,
                               research_question: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific research phase using appropriate meetings."""
        
        if phase == ResearchPhase.TEAM_SELECTION:
            return self._phase_team_selection(session_id, research_question, constraints)
        elif phase == ResearchPhase.PROJECT_SPECIFICATION:
            return self._phase_project_specification(session_id, research_question, constraints)
        elif phase == ResearchPhase.TOOLS_SELECTION:
            return self._phase_tools_selection(session_id, research_question, constraints)
        elif phase == ResearchPhase.TOOLS_IMPLEMENTATION:
            return self._phase_tools_implementation(session_id, research_question, constraints)
        elif phase == ResearchPhase.WORKFLOW_DESIGN:
            return self._phase_workflow_design(session_id, research_question, constraints)
        elif phase == ResearchPhase.EXECUTION:
            return self._phase_execution(session_id, research_question, constraints)
        elif phase == ResearchPhase.SYNTHESIS:
            return self._phase_synthesis(session_id, research_question, constraints)
        else:
            return {'success': False, 'error': f'Unknown phase: {phase}'}
    
    def _phase_team_selection(self, session_id: str, research_question: str, 
                             constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 1: Team Selection - Individual meeting with PI to define scientist agents."""
        
        agenda = MeetingAgenda(
            meeting_id=f"{session_id}_team_selection",
            meeting_type=MeetingType.INDIVIDUAL_MEETING,
            phase=ResearchPhase.TEAM_SELECTION,
            objectives=[
                "Analyze research problem complexity and scope",
                "Identify required domain expertise",
                "Determine optimal team composition",
                "Estimate resource requirements"
            ],
            participants=[self.pi_agent.agent_id],
            discussion_topics=[
                "Research problem analysis",
                "Expertise domain mapping",
                "Team size optimization",
                "Agent specialization requirements"
            ],
            expected_outcomes=[
                "List of required expertise domains",
                "Team composition plan",
                "Agent hiring strategy"
            ]
        )
        
        # Conduct individual meeting with PI
        meeting_result = self._conduct_individual_meeting(agenda, research_question, constraints)
        
        if meeting_result['success']:
            # Have PI analyze the problem and determine required expertise
            analysis_prompt = f"""
            As the Principal Investigator, analyze this research problem and determine the optimal team composition:
            
            Research Question: {research_question}
            Constraints: {constraints}
            
            Following the Virtual Lab approach, please provide:
            1. REQUIRED_EXPERTISE: List 3-6 specific expertise domains needed (e.g., biology, chemistry, computational_modeling, etc.)
            2. TEAM_SIZE: Optimal number of expert agents (2-8)
            3. PRIORITY_EXPERTS: Top 3 most critical expertise areas
            4. SPECIALIZATION_NOTES: Brief description of what each expert should focus on
            
            Format your response as:
            REQUIRED_EXPERTISE: [domain1, domain2, domain3, ...]
            TEAM_SIZE: [number]
            PRIORITY_EXPERTS: [domain1, domain2, domain3]
            SPECIALIZATION_NOTES: domain1: description | domain2: description | domain3: description
            """
            
            pi_response = self.pi_agent.generate_response(analysis_prompt, {
                'research_question': research_question,
                'constraints': constraints,
                'session_id': session_id
            })
            
            # Parse PI response
            team_plan = self._parse_team_selection_response(pi_response)
            
            # Hire agents based on PI's analysis
            if team_plan['required_expertise']:
                hiring_result = self.pi_agent.hire_agents(
                    self.agent_marketplace, 
                    team_plan['required_expertise'], 
                    constraints
                )
                
                return {
                    'success': True,
                    'meeting_record': meeting_result['meeting_record'],
                    'team_plan': team_plan,
                    'hired_agents': hiring_result,
                    'decisions': [
                        f"Team size: {team_plan['team_size']} experts",
                        f"Required expertise: {', '.join(team_plan['required_expertise'])}",
                        f"Successfully hired {len(hiring_result['hired_agents'])} agents"
                    ]
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to determine required expertise',
                    'meeting_record': meeting_result['meeting_record']
                }
        
        return meeting_result
    
    def _phase_project_specification(self, session_id: str, research_question: str,
                                   constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Project Specification - Team meeting to decide on key high-level details."""
        
        # Get hired agents from previous phase
        team_selection_result = self.phase_results.get('team_selection', {})
        hired_agents = team_selection_result.get('hired_agents', {}).get('hired_agents', {})
        
        if not hired_agents:
            return {'success': False, 'error': 'No agents available from team selection phase'}
        
        participants = [self.pi_agent.agent_id] + [agent.agent_id for agent in hired_agents.values()]
        
        agenda = MeetingAgenda(
            meeting_id=f"{session_id}_project_specification",
            meeting_type=MeetingType.TEAM_MEETING,
            phase=ResearchPhase.PROJECT_SPECIFICATION,
            objectives=[
                "Define specific project objectives and scope",
                "Establish success criteria and deliverables",
                "Agree on methodological approach",
                "Set timeline and milestones"
            ],
            participants=participants,
            discussion_topics=[
                "Project scope and boundaries",
                "Key research questions breakdown",
                "Success metrics definition",
                "Resource allocation strategy"
            ],
            expected_outcomes=[
                "Detailed project specification",
                "Agreed methodology framework",
                "Success criteria definition",
                "Timeline and milestone plan"
            ]
        )
        
        # Conduct team meeting
        meeting_result = self._conduct_team_meeting(agenda, hired_agents, research_question, constraints)
        
        if meeting_result['success']:
            # Extract project specification from meeting outcomes
            project_spec = self._extract_project_specification(meeting_result['meeting_record'])
            
            return {
                'success': True,
                'meeting_record': meeting_result['meeting_record'],
                'project_specification': project_spec,
                'decisions': meeting_result['meeting_record'].decisions
            }
        
        return meeting_result
    
    def _phase_tools_selection(self, session_id: str, research_question: str,
                              constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: Tools Selection - Team meeting to brainstorm computational tools."""
        
        # Get hired agents from team selection phase
        team_selection_result = self.phase_results.get('team_selection', {})
        hired_agents = team_selection_result.get('hired_agents', {}).get('hired_agents', {})
        
        if not hired_agents:
            return {'success': False, 'error': 'No agents available from team selection phase'}
        
        participants = [self.pi_agent.agent_id] + [agent.agent_id for agent in hired_agents.values()]
        
        agenda = MeetingAgenda(
            meeting_id=f"{session_id}_tools_selection",
            meeting_type=MeetingType.TEAM_MEETING,
            phase=ResearchPhase.TOOLS_SELECTION,
            objectives=[
                "Brainstorm computational and analytical tools",
                "Evaluate tool suitability for research objectives",
                "Select optimal tool combination",
                "Plan tool integration strategy"
            ],
            participants=participants,
            discussion_topics=[
                "Available computational tools review",
                "Tool capability assessment",
                "Integration complexity evaluation",
                "Alternative tool considerations"
            ],
            expected_outcomes=[
                "Selected tool suite",
                "Tool integration plan",
                "Implementation priority order",
                "Resource requirements per tool"
            ]
        )
        
        # Conduct team meeting focused on tool selection
        meeting_result = self._conduct_team_meeting(agenda, hired_agents, research_question, constraints)
        
        if meeting_result['success']:
            # Extract tool selection from meeting outcomes
            tool_selection = self._extract_tool_selection(meeting_result['meeting_record'])
            
            return {
                'success': True,
                'meeting_record': meeting_result['meeting_record'],
                'selected_tools': tool_selection,
                'decisions': meeting_result['meeting_record'].decisions
            }
        
        return meeting_result
    
    def _phase_tools_implementation(self, session_id: str, research_question: str,
                                   constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 4: Tools Implementation - Series of individual meetings to implement selected tools."""
        
        # Get selected tools from previous phase
        tools_selection_result = self.phase_results.get('tools_selection', {})
        selected_tools = tools_selection_result.get('selected_tools', {})
        
        if not selected_tools:
            return {'success': False, 'error': 'No tools selected in previous phase'}
        
        # Get hired agents
        team_selection_result = self.phase_results.get('team_selection', {})
        hired_agents = team_selection_result.get('hired_agents', {}).get('hired_agents', {})
        
        implementation_results = {}
        
        # For each selected tool, conduct implementation meetings
        for tool_name, tool_details in selected_tools.items():
            # Find best agent for implementing this tool
            best_agent = self._select_agent_for_tool(tool_name, tool_details, hired_agents)
            
            if not best_agent:
                logger.warning(f"No suitable agent found for tool: {tool_name}")
                continue
            
            # Conduct individual meeting for tool implementation
            implementation_agenda = MeetingAgenda(
                meeting_id=f"{session_id}_implement_{tool_name}",
                meeting_type=MeetingType.INDIVIDUAL_MEETING,
                phase=ResearchPhase.TOOLS_IMPLEMENTATION,
                objectives=[
                    f"Implement {tool_name} for research objectives",
                    "Develop tool integration interface",
                    "Create usage documentation",
                    "Test tool functionality"
                ],
                participants=[best_agent.agent_id, self.scientific_critic.agent_id],
                discussion_topics=[
                    f"{tool_name} implementation strategy",
                    "Integration requirements",
                    "Testing and validation approach",
                    "Documentation standards"
                ],
                expected_outcomes=[
                    f"Functional {tool_name} implementation",
                    "Integration interface",
                    "Usage documentation",
                    "Test results"
                ]
            )
            
            # Implementation meeting with agent and scientific critic
            impl_result = self._conduct_implementation_meeting(
                implementation_agenda, best_agent, tool_name, tool_details
            )
            
            implementation_results[tool_name] = impl_result
        
        # Aggregate implementation results
        all_successful = all(result.get('success', False) for result in implementation_results.values())
        
        return {
            'success': all_successful,
            'implementation_results': implementation_results,
            'implemented_tools': list(implementation_results.keys()),
            'decisions': [f"Implemented {tool}" for tool in implementation_results.keys()]
        }
    
    def _phase_workflow_design(self, session_id: str, research_question: str,
                              constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 5: Workflow Design - Individual meeting with PI to determine workflow."""
        
        # Get implementation results
        implementation_result = self.phase_results.get('tools_implementation', {})
        implemented_tools = implementation_result.get('implemented_tools', [])
        
        if not implemented_tools:
            return {'success': False, 'error': 'No tools implemented to create workflow'}
        
        agenda = MeetingAgenda(
            meeting_id=f"{session_id}_workflow_design",
            meeting_type=MeetingType.INDIVIDUAL_MEETING,
            phase=ResearchPhase.WORKFLOW_DESIGN,
            objectives=[
                "Design integrated workflow using implemented tools",
                "Define data flow between tools",
                "Establish quality control checkpoints",
                "Create execution plan"
            ],
            participants=[self.pi_agent.agent_id],
            discussion_topics=[
                "Tool integration sequence",
                "Data pipeline design",
                "Quality control measures",
                "Error handling strategies"
            ],
            expected_outcomes=[
                "Complete workflow design",
                "Execution plan",
                "Quality control framework",
                "Resource requirements"
            ]
        )
        
        # Conduct individual meeting with PI for workflow design
        meeting_result = self._conduct_individual_meeting(agenda, research_question, constraints)
        
        if meeting_result['success']:
            # Have PI design the workflow
            workflow_prompt = f"""
            As the Principal Investigator, design a comprehensive research workflow using the implemented tools:
            
            Research Question: {research_question}
            Available Tools: {implemented_tools}
            Implementation Results: {implementation_result}
            
            Design a workflow that:
            1. WORKFLOW_STEPS: Sequential steps using the available tools
            2. DATA_FLOW: How data flows between tools
            3. QUALITY_CONTROL: Checkpoints and validation steps
            4. EXECUTION_ORDER: Optimal order of tool execution
            5. SUCCESS_METRICS: How to measure workflow success
            
            Format as:
            WORKFLOW_STEPS: step1: description | step2: description | step3: description
            DATA_FLOW: tool1 -> tool2 -> tool3
            QUALITY_CONTROL: checkpoint1 | checkpoint2 | checkpoint3
            EXECUTION_ORDER: [tool1, tool2, tool3]
            SUCCESS_METRICS: metric1 | metric2 | metric3
            """
            
            pi_response = self.pi_agent.generate_response(workflow_prompt, {
                'research_question': research_question,
                'implemented_tools': implemented_tools,
                'session_id': session_id
            })
            
            # Parse workflow design
            workflow_design = self._parse_workflow_design_response(pi_response)
            
            return {
                'success': True,
                'meeting_record': meeting_result['meeting_record'],
                'workflow_design': workflow_design,
                'decisions': [
                    f"Designed workflow with {len(workflow_design.get('workflow_steps', []))} steps",
                    f"Integrated {len(implemented_tools)} tools",
                    f"Established {len(workflow_design.get('quality_control', []))} quality checkpoints"
                ]
            }
        
        return meeting_result
    
    def _phase_execution(self, session_id: str, research_question: str,
                        constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 6: Execution - Execute the designed workflow with agent collaboration."""
        
        # Get workflow design
        workflow_result = self.phase_results.get('workflow_design', {})
        workflow_design = workflow_result.get('workflow_design', {})
        
        if not workflow_design:
            return {'success': False, 'error': 'No workflow design available'}
        
        # Get hired agents
        team_selection_result = self.phase_results.get('team_selection', {})
        hired_agents = team_selection_result.get('hired_agents', {}).get('hired_agents', {})
        
        execution_results = {}
        
        # Execute workflow steps
        workflow_steps = workflow_design.get('workflow_steps', [])
        execution_order = workflow_design.get('execution_order', [])
        
        for step_name in execution_order:
            if step_name in workflow_steps:
                step_description = workflow_steps[step_name]
                
                # Find appropriate agent for this step
                step_agent = self._select_agent_for_step(step_name, step_description, hired_agents)
                
                if step_agent:
                    # Execute step with agent
                    step_result = self._execute_workflow_step(
                        step_name, step_description, step_agent, session_id
                    )
                    execution_results[step_name] = step_result
                else:
                    logger.warning(f"No suitable agent found for step: {step_name}")
                    execution_results[step_name] = {
                        'success': False,
                        'error': f'No agent available for step: {step_name}'
                    }
        
        # Facilitate cross-agent interaction and critique
        cross_interaction_result = self._facilitate_cross_agent_interaction(
            session_id, hired_agents, execution_results
        )
        
        # Aggregate results
        all_successful = all(result.get('success', False) for result in execution_results.values())
        
        return {
            'success': all_successful,
            'execution_results': execution_results,
            'cross_interaction': cross_interaction_result,
            'completed_steps': list(execution_results.keys()),
            'decisions': [f"Executed step: {step}" for step in execution_results.keys()]
        }
    
    def _phase_synthesis(self, session_id: str, research_question: str,
                        constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 7: Synthesis - Final synthesis and critique of results."""
        
        # Get all previous phase results
        execution_result = self.phase_results.get('execution', {})
        
        if not execution_result.get('success', False):
            return {'success': False, 'error': 'Execution phase failed, cannot synthesize'}
        
        # Conduct final synthesis meeting with all agents
        team_selection_result = self.phase_results.get('team_selection', {})
        hired_agents = team_selection_result.get('hired_agents', {}).get('hired_agents', {})
        
        participants = [self.pi_agent.agent_id, self.scientific_critic.agent_id] + \
                      [agent.agent_id for agent in hired_agents.values()]
        
        agenda = MeetingAgenda(
            meeting_id=f"{session_id}_synthesis",
            meeting_type=MeetingType.TEAM_MEETING,
            phase=ResearchPhase.SYNTHESIS,
            objectives=[
                "Synthesize findings from all research phases",
                "Conduct comprehensive scientific critique",
                "Identify key insights and discoveries",
                "Formulate conclusions and recommendations"
            ],
            participants=participants,
            discussion_topics=[
                "Key findings compilation",
                "Cross-domain insights",
                "Methodological validation",
                "Future research directions"
            ],
            expected_outcomes=[
                "Comprehensive research synthesis",
                "Scientific critique and validation",
                "Key discoveries and insights",
                "Research recommendations"
            ]
        )
        
        # Conduct final synthesis meeting
        meeting_result = self._conduct_team_meeting(agenda, hired_agents, research_question, constraints)
        
        if meeting_result['success']:
            # Generate comprehensive scientific critique
            critique_result = self.scientific_critic.critique_research_output(
                output_content=str(meeting_result['meeting_record'].outcomes),
                output_type="virtual_lab_research",
                context={'session_id': session_id, 'research_question': research_question}
            )
            
            # Extract final synthesis
            final_synthesis = self._extract_final_synthesis(meeting_result['meeting_record'], critique_result)
            
            return {
                'success': True,
                'meeting_record': meeting_result['meeting_record'],
                'synthesis': final_synthesis,
                'critique': critique_result,
                'decisions': meeting_result['meeting_record'].decisions
            }
        
        return meeting_result
    
    def _conduct_team_meeting(self, agenda: MeetingAgenda, hired_agents: Dict[str, BaseAgent],
                             research_question: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct a team meeting with multiple agents."""
        
        logger.info(f"Starting team meeting: {agenda.meeting_id}")
        start_time = time.time()
        
        # Initialize meeting record
        discussion_transcript = []
        outcomes = {}
        decisions = []
        action_items = []
        
        try:
            # PI opens the meeting
            opening_prompt = f"""
            You are facilitating a team meeting for: {agenda.phase.value}
            
            Research Question: {research_question}
            Meeting Objectives: {', '.join(agenda.objectives)}
            Discussion Topics: {', '.join(agenda.discussion_topics)}
            
            Please open the meeting by:
            1. Stating the objectives
            2. Outlining the discussion topics
            3. Inviting each expert to share their perspective
            
            Keep your opening brief and focused.
            """
            
            pi_opening = self.pi_agent.generate_response(opening_prompt, {
                'agenda': agenda.__dict__,
                'research_question': research_question
            })
            
            discussion_transcript.append({
                'speaker': self.pi_agent.agent_id,
                'role': 'Principal Investigator',
                'message': pi_opening,
                'timestamp': time.time()
            })
            
            # Each agent contributes based on their expertise
            for expertise, agent in hired_agents.items():
                contribution_prompt = f"""
                In this team meeting about {agenda.phase.value}, please contribute your expertise in {expertise}.
                
                Research Question: {research_question}
                Discussion Topics: {', '.join(agenda.discussion_topics)}
                
                PI's Opening: {pi_opening}
                
                Provide your expert perspective on:
                1. How your expertise applies to this research
                2. Specific insights or recommendations
                3. Potential challenges or considerations
                4. Suggested approaches or methods
                
                Be specific and actionable in your contribution.
                """
                
                agent_contribution = agent.generate_response(contribution_prompt, {
                    'meeting_context': agenda.__dict__,
                    'research_question': research_question,
                    'expertise': expertise
                })
                
                discussion_transcript.append({
                    'speaker': agent.agent_id,
                    'role': agent.role,
                    'expertise': expertise,
                    'message': agent_contribution,
                    'timestamp': time.time()
                })
            
            # PI synthesizes the discussion
            synthesis_prompt = f"""
            As the Principal Investigator, synthesize the expert contributions and make decisions:
            
            Expert Contributions:
            {self._format_contributions_for_synthesis(discussion_transcript[1:])}
            
            Based on these expert inputs, please:
            1. OUTCOMES: Key outcomes achieved from this meeting
            2. DECISIONS: Specific decisions made
            3. ACTION_ITEMS: Next steps to be taken
            
            Format as:
            OUTCOMES: outcome1 | outcome2 | outcome3
            DECISIONS: decision1 | decision2 | decision3
            ACTION_ITEMS: action1 | action2 | action3
            """
            
            pi_synthesis = self.pi_agent.generate_response(synthesis_prompt, {
                'discussion_transcript': discussion_transcript,
                'agenda': agenda.__dict__
            })
            
            discussion_transcript.append({
                'speaker': self.pi_agent.agent_id,
                'role': 'Principal Investigator',
                'message': pi_synthesis,
                'timestamp': time.time()
            })
            
            # Parse synthesis results
            outcomes, decisions, action_items = self._parse_meeting_synthesis(pi_synthesis)
            
            # Create meeting record
            meeting_record = MeetingRecord(
                meeting_id=agenda.meeting_id,
                meeting_type=agenda.meeting_type,
                phase=agenda.phase,
                participants=agenda.participants,
                agenda=agenda,
                discussion_transcript=discussion_transcript,
                outcomes=outcomes,
                decisions=decisions,
                action_items=action_items,
                start_time=start_time,
                end_time=time.time(),
                success=True
            )
            
            # Store meeting
            self.meeting_history.append(meeting_record)
            
            logger.info(f"Team meeting {agenda.meeting_id} completed successfully")
            
            return {
                'success': True,
                'meeting_record': meeting_record,
                'duration': meeting_record.end_time - meeting_record.start_time
            }
            
        except Exception as e:
            logger.error(f"Team meeting {agenda.meeting_id} failed: {e}")
            
            # Create failed meeting record
            meeting_record = MeetingRecord(
                meeting_id=agenda.meeting_id,
                meeting_type=agenda.meeting_type,
                phase=agenda.phase,
                participants=agenda.participants,
                agenda=agenda,
                discussion_transcript=discussion_transcript,
                outcomes={'error': str(e)},
                decisions=[],
                action_items=[],
                start_time=start_time,
                end_time=time.time(),
                success=False
            )
            
            self.meeting_history.append(meeting_record)
            
            return {
                'success': False,
                'error': str(e),
                'meeting_record': meeting_record
            }
    
    def _conduct_individual_meeting(self, agenda: MeetingAgenda, research_question: str,
                                   constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct an individual meeting (usually with PI)."""
        
        logger.info(f"Starting individual meeting: {agenda.meeting_id}")
        start_time = time.time()
        
        discussion_transcript = []
        outcomes = {}
        decisions = []
        action_items = []
        
        try:
            # Conduct individual meeting with PI
            individual_prompt = f"""
            You are conducting an individual meeting for: {agenda.phase.value}
            
            Research Question: {research_question}
            Objectives: {', '.join(agenda.objectives)}
            Expected Outcomes: {', '.join(agenda.expected_outcomes)}
            Constraints: {constraints}
            
            Please analyze and provide:
            1. Detailed analysis of the research requirements
            2. Specific recommendations and decisions
            3. Action items for next steps
            
            Be comprehensive and decisive in your analysis.
            """
            
            pi_response = self.pi_agent.generate_response(individual_prompt, {
                'agenda': agenda.__dict__,
                'research_question': research_question,
                'constraints': constraints
            })
            
            discussion_transcript.append({
                'speaker': self.pi_agent.agent_id,
                'role': 'Principal Investigator',
                'message': pi_response,
                'timestamp': time.time()
            })
            
            # Parse response for outcomes, decisions, and action items
            outcomes = {'analysis': pi_response}
            decisions = [f"Completed {agenda.phase.value} analysis"]
            action_items = [f"Proceed to next phase based on {agenda.phase.value} results"]
            
            # Create meeting record
            meeting_record = MeetingRecord(
                meeting_id=agenda.meeting_id,
                meeting_type=agenda.meeting_type,
                phase=agenda.phase,
                participants=agenda.participants,
                agenda=agenda,
                discussion_transcript=discussion_transcript,
                outcomes=outcomes,
                decisions=decisions,
                action_items=action_items,
                start_time=start_time,
                end_time=time.time(),
                success=True
            )
            
            # Store meeting
            self.meeting_history.append(meeting_record)
            
            logger.info(f"Individual meeting {agenda.meeting_id} completed successfully")
            
            return {
                'success': True,
                'meeting_record': meeting_record,
                'duration': meeting_record.end_time - meeting_record.start_time
            }
            
        except Exception as e:
            logger.error(f"Individual meeting {agenda.meeting_id} failed: {e}")
            
            meeting_record = MeetingRecord(
                meeting_id=agenda.meeting_id,
                meeting_type=agenda.meeting_type,
                phase=agenda.phase,
                participants=agenda.participants,
                agenda=agenda,
                discussion_transcript=discussion_transcript,
                outcomes={'error': str(e)},
                decisions=[],
                action_items=[],
                start_time=start_time,
                end_time=time.time(),
                success=False
            )
            
            self.meeting_history.append(meeting_record)
            
            return {
                'success': False,
                'error': str(e),
                'meeting_record': meeting_record
            }
    
    def _conduct_implementation_meeting(self, agenda: MeetingAgenda, agent: BaseAgent,
                                       tool_name: str, tool_details: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct an implementation meeting for a specific tool."""
        
        logger.info(f"Starting implementation meeting for {tool_name}")
        start_time = time.time()
        
        discussion_transcript = []
        outcomes = {}
        decisions = []
        action_items = []
        
        try:
            # Agent implements the tool
            implementation_prompt = f"""
            You are implementing {tool_name} for the research project.
            
            Tool Details: {tool_details}
            Implementation Requirements: {agenda.objectives}
            
            Please provide:
            1. IMPLEMENTATION_PLAN: Step-by-step implementation approach
            2. CODE_FRAMEWORK: Basic code structure or framework
            3. INTEGRATION_POINTS: How this tool connects with others
            4. TESTING_STRATEGY: How to validate the implementation
            5. DOCUMENTATION: Key usage information
            
            Format as:
            IMPLEMENTATION_PLAN: step1 | step2 | step3
            CODE_FRAMEWORK: [basic code structure]
            INTEGRATION_POINTS: point1 | point2 | point3
            TESTING_STRATEGY: test1 | test2 | test3
            DOCUMENTATION: usage_info
            """
            
            agent_response = agent.generate_response(implementation_prompt, {
                'tool_name': tool_name,
                'tool_details': tool_details,
                'agenda': agenda.__dict__
            })
            
            discussion_transcript.append({
                'speaker': agent.agent_id,
                'role': agent.role,
                'message': agent_response,
                'timestamp': time.time()
            })
            
            # Scientific critic reviews the implementation
            critique_prompt = f"""
            Review this tool implementation for scientific rigor and technical quality:
            
            Tool: {tool_name}
            Implementation: {agent_response}
            
            Provide:
            1. QUALITY_ASSESSMENT: Overall quality score (1-10)
            2. STRENGTHS: Key strengths of the implementation
            3. WEAKNESSES: Areas for improvement
            4. RECOMMENDATIONS: Specific improvement suggestions
            
            Format as:
            QUALITY_ASSESSMENT: [score]
            STRENGTHS: strength1 | strength2 | strength3
            WEAKNESSES: weakness1 | weakness2 | weakness3
            RECOMMENDATIONS: rec1 | rec2 | rec3
            """
            
            critic_response = self.scientific_critic.generate_response(critique_prompt, {
                'tool_name': tool_name,
                'implementation': agent_response
            })
            
            discussion_transcript.append({
                'speaker': self.scientific_critic.agent_id,
                'role': 'Scientific Critic',
                'message': critic_response,
                'timestamp': time.time()
            })
            
            # Parse implementation details
            implementation_details = self._parse_implementation_response(agent_response)
            critique_details = self._parse_critique_response(critic_response)
            
            outcomes = {
                'implementation': implementation_details,
                'critique': critique_details,
                'tool_name': tool_name
            }
            
            decisions = [f"Implemented {tool_name}", f"Quality score: {critique_details.get('quality_score', 'N/A')}"]
            action_items = [f"Integrate {tool_name} into workflow"]
            
            # Create meeting record
            meeting_record = MeetingRecord(
                meeting_id=agenda.meeting_id,
                meeting_type=agenda.meeting_type,
                phase=agenda.phase,
                participants=agenda.participants,
                agenda=agenda,
                discussion_transcript=discussion_transcript,
                outcomes=outcomes,
                decisions=decisions,
                action_items=action_items,
                start_time=start_time,
                end_time=time.time(),
                success=True
            )
            
            # Store meeting
            self.meeting_history.append(meeting_record)
            
            return {
                'success': True,
                'meeting_record': meeting_record,
                'implementation_details': implementation_details,
                'critique': critique_details
            }
            
        except Exception as e:
            logger.error(f"Implementation meeting for {tool_name} failed: {e}")
            
            meeting_record = MeetingRecord(
                meeting_id=agenda.meeting_id,
                meeting_type=agenda.meeting_type,
                phase=agenda.phase,
                participants=agenda.participants,
                agenda=agenda,
                discussion_transcript=discussion_transcript,
                outcomes={'error': str(e)},
                decisions=[],
                action_items=[],
                start_time=start_time,
                end_time=time.time(),
                success=False
            )
            
            self.meeting_history.append(meeting_record)
            
            return {
                'success': False,
                'error': str(e),
                'meeting_record': meeting_record
            }
    
    def _facilitate_cross_agent_interaction(self, session_id: str, hired_agents: Dict[str, BaseAgent],
                                          execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Facilitate cross-agent interaction for cross-pollination of ideas."""
        
        logger.info(f"Facilitating cross-agent interaction for session {session_id}")
        
        cross_interactions = []
        synthesis_points = []
        
        # Create summary of execution results for cross-analysis
        results_summary = []
        for step_name, result in execution_results.items():
            if result.get('success', False):
                results_summary.append(f"{step_name}: {str(result.get('output', ''))[:200]}...")
        
        if len(results_summary) < 2:
            return {'interactions': cross_interactions, 'synthesis_points': synthesis_points}
        
        # Have each agent comment on others' findings
        for expertise, agent in hired_agents.items():
            try:
                # Filter out this agent's own results
                other_results = [summary for summary in results_summary 
                               if not summary.startswith(agent.agent_id)]
                
                if other_results:
                    cross_prompt = f"""
                    Based on your expertise in {expertise}, analyze these findings from other experts:
                    
                    {chr(10).join(other_results)}
                    
                    Provide:
                    1. CONNECTIONS: How these findings connect to your expertise
                    2. CONTRADICTIONS: Any contradictions or inconsistencies you observe
                    3. SYNERGIES: Potential synergies between different findings
                    4. INSIGHTS: New insights from cross-domain perspective
                    
                    Format as:
                    CONNECTIONS: connection1 | connection2 | connection3
                    CONTRADICTIONS: contradiction1 | contradiction2
                    SYNERGIES: synergy1 | synergy2 | synergy3
                    INSIGHTS: insight1 | insight2 | insight3
                    """
                    
                    cross_response = agent.generate_response(cross_prompt, {
                        'session_id': session_id,
                        'interaction_type': 'cross_pollination',
                        'other_results': other_results
                    })
                    
                    cross_interactions.append({
                        'agent_id': agent.agent_id,
                        'expertise': expertise,
                        'cross_analysis': cross_response,
                        'timestamp': time.time()
                    })
                    
            except Exception as e:
                logger.warning(f"Cross-pollination failed for agent {agent.agent_id}: {e}")
        
        # Extract synthesis points from cross-interactions
        for interaction in cross_interactions:
            synthesis_points.extend(self._extract_synthesis_points(interaction['cross_analysis']))
        
        return {
            'interactions': cross_interactions,
            'synthesis_points': synthesis_points
        }
    
    # Helper methods for parsing responses
    
    def _parse_team_selection_response(self, response: str) -> Dict[str, Any]:
        """Parse team selection response from PI."""
        import re
        
        result = {
            'required_expertise': [],
            'team_size': 3,
            'priority_experts': [],
            'specialization_notes': {}
        }
        
        try:
            # Extract required expertise
            expertise_match = re.search(r'REQUIRED_EXPERTISE:\s*\[([^\]]+)\]', response)
            if expertise_match:
                expertise_str = expertise_match.group(1)
                result['required_expertise'] = [e.strip().strip('"\'') for e in expertise_str.split(',')]
            
            # Extract team size
            size_match = re.search(r'TEAM_SIZE:\s*(\d+)', response)
            if size_match:
                result['team_size'] = int(size_match.group(1))
            
            # Extract priority experts
            priority_match = re.search(r'PRIORITY_EXPERTS:\s*\[([^\]]+)\]', response)
            if priority_match:
                priority_str = priority_match.group(1)
                result['priority_experts'] = [p.strip().strip('"\'') for p in priority_str.split(',')]
            
            # Extract specialization notes
            notes_match = re.search(r'SPECIALIZATION_NOTES:\s*([^\n]+)', response)
            if notes_match:
                notes_str = notes_match.group(1)
                for note in notes_str.split('|'):
                    if ':' in note:
                        domain, description = note.split(':', 1)
                        result['specialization_notes'][domain.strip()] = description.strip()
        
        except Exception as e:
            logger.warning(f"Failed to parse team selection response: {e}")
        
        return result
    
    def _parse_workflow_design_response(self, response: str) -> Dict[str, Any]:
        """Parse workflow design response."""
        import re
        
        result = {
            'workflow_steps': {},
            'data_flow': [],
            'quality_control': [],
            'execution_order': [],
            'success_metrics': []
        }
        
        try:
            # Extract workflow steps
            steps_match = re.search(r'WORKFLOW_STEPS:\s*([^\n]+)', response)
            if steps_match:
                steps_str = steps_match.group(1)
                for step in steps_str.split('|'):
                    if ':' in step:
                        step_name, description = step.split(':', 1)
                        result['workflow_steps'][step_name.strip()] = description.strip()
            
            # Extract data flow
            flow_match = re.search(r'DATA_FLOW:\s*([^\n]+)', response)
            if flow_match:
                flow_str = flow_match.group(1)
                result['data_flow'] = [f.strip() for f in flow_str.split('->')]
            
            # Extract quality control
            qc_match = re.search(r'QUALITY_CONTROL:\s*([^\n]+)', response)
            if qc_match:
                qc_str = qc_match.group(1)
                result['quality_control'] = [q.strip() for q in qc_str.split('|')]
            
            # Extract execution order
            order_match = re.search(r'EXECUTION_ORDER:\s*\[([^\]]+)\]', response)
            if order_match:
                order_str = order_match.group(1)
                result['execution_order'] = [o.strip().strip('"\'') for o in order_str.split(',')]
            
            # Extract success metrics
            metrics_match = re.search(r'SUCCESS_METRICS:\s*([^\n]+)', response)
            if metrics_match:
                metrics_str = metrics_match.group(1)
                result['success_metrics'] = [m.strip() for m in metrics_str.split('|')]
        
        except Exception as e:
            logger.warning(f"Failed to parse workflow design response: {e}")
        
        return result
    
    def _parse_meeting_synthesis(self, synthesis: str) -> Tuple[Dict[str, Any], List[str], List[str]]:
        """Parse meeting synthesis into outcomes, decisions, and action items."""
        import re
        
        outcomes = {}
        decisions = []
        action_items = []
        
        try:
            # Extract outcomes
            outcomes_match = re.search(r'OUTCOMES:\s*([^\n]+)', synthesis)
            if outcomes_match:
                outcomes_str = outcomes_match.group(1)
                outcomes_list = [o.strip() for o in outcomes_str.split('|')]
                outcomes = {f'outcome_{i+1}': outcome for i, outcome in enumerate(outcomes_list)}
            
            # Extract decisions
            decisions_match = re.search(r'DECISIONS:\s*([^\n]+)', synthesis)
            if decisions_match:
                decisions_str = decisions_match.group(1)
                decisions = [d.strip() for d in decisions_str.split('|')]
            
            # Extract action items
            actions_match = re.search(r'ACTION_ITEMS:\s*([^\n]+)', synthesis)
            if actions_match:
                actions_str = actions_match.group(1)
                action_items = [a.strip() for a in actions_str.split('|')]
        
        except Exception as e:
            logger.warning(f"Failed to parse meeting synthesis: {e}")
        
        return outcomes, decisions, action_items
    
    def _parse_implementation_response(self, response: str) -> Dict[str, Any]:
        """Parse implementation response."""
        import re
        
        result = {
            'implementation_plan': [],
            'code_framework': '',
            'integration_points': [],
            'testing_strategy': [],
            'documentation': ''
        }
        
        try:
            # Extract implementation plan
            plan_match = re.search(r'IMPLEMENTATION_PLAN:\s*([^\n]+)', response)
            if plan_match:
                plan_str = plan_match.group(1)
                result['implementation_plan'] = [p.strip() for p in plan_str.split('|')]
            
            # Extract code framework
            code_match = re.search(r'CODE_FRAMEWORK:\s*\[([^\]]+)\]', response)
            if code_match:
                result['code_framework'] = code_match.group(1).strip()
            
            # Extract integration points
            integration_match = re.search(r'INTEGRATION_POINTS:\s*([^\n]+)', response)
            if integration_match:
                integration_str = integration_match.group(1)
                result['integration_points'] = [i.strip() for i in integration_str.split('|')]
            
            # Extract testing strategy
            testing_match = re.search(r'TESTING_STRATEGY:\s*([^\n]+)', response)
            if testing_match:
                testing_str = testing_match.group(1)
                result['testing_strategy'] = [t.strip() for t in testing_str.split('|')]
            
            # Extract documentation
            doc_match = re.search(r'DOCUMENTATION:\s*([^\n]+)', response)
            if doc_match:
                result['documentation'] = doc_match.group(1).strip()
        
        except Exception as e:
            logger.warning(f"Failed to parse implementation response: {e}")
        
        return result
    
    def _parse_critique_response(self, response: str) -> Dict[str, Any]:
        """Parse critique response."""
        import re
        
        result = {
            'quality_score': 5,
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        try:
            # Extract quality score
            score_match = re.search(r'QUALITY_ASSESSMENT:\s*(\d+)', response)
            if score_match:
                result['quality_score'] = int(score_match.group(1))
            
            # Extract strengths
            strengths_match = re.search(r'STRENGTHS:\s*([^\n]+)', response)
            if strengths_match:
                strengths_str = strengths_match.group(1)
                result['strengths'] = [s.strip() for s in strengths_str.split('|')]
            
            # Extract weaknesses
            weaknesses_match = re.search(r'WEAKNESSES:\s*([^\n]+)', response)
            if weaknesses_match:
                weaknesses_str = weaknesses_match.group(1)
                result['weaknesses'] = [w.strip() for w in weaknesses_str.split('|')]
            
            # Extract recommendations
            recs_match = re.search(r'RECOMMENDATIONS:\s*([^\n]+)', response)
            if recs_match:
                recs_str = recs_match.group(1)
                result['recommendations'] = [r.strip() for r in recs_str.split('|')]
        
        except Exception as e:
            logger.warning(f"Failed to parse critique response: {e}")
        
        return result
    
    # Additional helper methods
    
    def _format_contributions_for_synthesis(self, contributions: List[Dict[str, Any]]) -> str:
        """Format agent contributions for synthesis."""
        formatted = []
        for contrib in contributions:
            expertise = contrib.get('expertise', 'Unknown')
            message = contrib.get('message', '')
            formatted.append(f"{expertise}: {message[:300]}{'...' if len(message) > 300 else ''}")
        return '\n\n'.join(formatted)
    
    def _extract_project_specification(self, meeting_record: MeetingRecord) -> Dict[str, Any]:
        """Extract project specification from meeting outcomes."""
        outcomes = meeting_record.outcomes
        return {
            'objectives': outcomes.get('objectives', []),
            'scope': outcomes.get('scope', ''),
            'methodology': outcomes.get('methodology', ''),
            'success_criteria': outcomes.get('success_criteria', []),
            'timeline': outcomes.get('timeline', {})
        }
    
    def _extract_tool_selection(self, meeting_record: MeetingRecord) -> Dict[str, Any]:
        """Extract tool selection from meeting outcomes."""
        outcomes = meeting_record.outcomes
        return {
            'selected_tools': outcomes.get('selected_tools', {}),
            'integration_plan': outcomes.get('integration_plan', ''),
            'priority_order': outcomes.get('priority_order', []),
            'resource_requirements': outcomes.get('resource_requirements', {})
        }
    
    def _extract_final_synthesis(self, meeting_record: MeetingRecord, 
                                critique_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract final synthesis from meeting and critique."""
        return {
            'key_findings': meeting_record.outcomes.get('key_findings', []),
            'insights': meeting_record.outcomes.get('insights', []),
            'conclusions': meeting_record.outcomes.get('conclusions', []),
            'recommendations': meeting_record.outcomes.get('recommendations', []),
            'critique_score': critique_result.get('overall_score', 0),
            'critique_summary': critique_result.get('summary', ''),
            'validated_findings': critique_result.get('validated_findings', [])
        }
    
    def _extract_synthesis_points(self, cross_analysis: str) -> List[str]:
        """Extract synthesis points from cross-agent analysis."""
        import re
        
        synthesis_points = []
        
        try:
            # Extract insights
            insights_match = re.search(r'INSIGHTS:\s*([^\n]+)', cross_analysis)
            if insights_match:
                insights_str = insights_match.group(1)
                synthesis_points.extend([i.strip() for i in insights_str.split('|')])
            
            # Extract synergies
            synergies_match = re.search(r'SYNERGIES:\s*([^\n]+)', cross_analysis)
            if synergies_match:
                synergies_str = synergies_match.group(1)
                synthesis_points.extend([s.strip() for s in synergies_str.split('|')])
        
        except Exception as e:
            logger.warning(f"Failed to extract synthesis points: {e}")
        
        return synthesis_points
    
    def _select_agent_for_tool(self, tool_name: str, tool_details: Dict[str, Any], 
                              hired_agents: Dict[str, BaseAgent]) -> Optional[BaseAgent]:
        """Select the best agent for implementing a specific tool."""
        best_agent = None
        best_score = 0.0
        
        for expertise, agent in hired_agents.items():
            relevance_score = agent.assess_task_relevance(f"Implement {tool_name} tool")
            if relevance_score > best_score:
                best_score = relevance_score
                best_agent = agent
        
        return best_agent
    
    def _select_agent_for_step(self, step_name: str, step_description: str,
                              hired_agents: Dict[str, BaseAgent]) -> Optional[BaseAgent]:
        """Select the best agent for executing a workflow step."""
        best_agent = None
        best_score = 0.0
        
        for expertise, agent in hired_agents.items():
            relevance_score = agent.assess_task_relevance(f"{step_name}: {step_description}")
            if relevance_score > best_score:
                best_score = relevance_score
                best_agent = agent
        
        return best_agent
    
    def _execute_workflow_step(self, step_name: str, step_description: str,
                              agent: BaseAgent, session_id: str) -> Dict[str, Any]:
        """Execute a single workflow step with an agent."""
        try:
            step_prompt = f"""
            Execute this workflow step: {step_name}
            
            Description: {step_description}
            Session ID: {session_id}
            
            Please:
            1. Perform the required analysis or computation
            2. Provide detailed results
            3. Note any issues or considerations
            4. Suggest improvements or next steps
            
            Be specific and thorough in your execution.
            """
            
            step_result = agent.generate_response(step_prompt, {
                'step_name': step_name,
                'step_description': step_description,
                'session_id': session_id
            })
            
            return {
                'success': True,
                'step_name': step_name,
                'agent_id': agent.agent_id,
                'output': step_result,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to execute step {step_name}: {e}")
            return {
                'success': False,
                'step_name': step_name,
                'agent_id': agent.agent_id,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _compile_final_results(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compile final results from all phases."""
        final_results = {
            'session_summary': {
                'session_id': session_data['session_id'],
                'research_question': session_data['research_question'],
                'duration': session_data.get('duration', 0),
                'phases_completed': len(session_data.get('phases', {})),
                'total_meetings': len([phase for phase in session_data.get('phases', {}).values() 
                                    if phase.get('meeting_record')])
            },
            'key_outcomes': {},
            'validated_findings': [],
            'research_recommendations': [],
            'quality_assessment': {}
        }
        
        # Extract key outcomes from each phase
        for phase_name, phase_result in session_data.get('phases', {}).items():
            if phase_result.get('success', False):
                final_results['key_outcomes'][phase_name] = phase_result.get('decisions', [])
        
        # Get synthesis results if available
        synthesis_result = session_data.get('phases', {}).get('synthesis', {})
        if synthesis_result.get('success', False):
            synthesis_data = synthesis_result.get('synthesis', {})
            final_results['validated_findings'] = synthesis_data.get('key_findings', [])
            final_results['research_recommendations'] = synthesis_data.get('recommendations', [])
            
            critique_data = synthesis_result.get('critique', {})
            final_results['quality_assessment'] = {
                'overall_score': critique_data.get('overall_score', 0),
                'summary': critique_data.get('summary', ''),
                'strengths': critique_data.get('strengths', []),
                'weaknesses': critique_data.get('weaknesses', [])
            }
        
        return final_results
    
    # Public methods for accessing meeting system state
    
    def get_meeting_history(self, limit: Optional[int] = None) -> List[MeetingRecord]:
        """Get meeting history."""
        if limit:
            return self.meeting_history[-limit:]
        return self.meeting_history.copy()
    
    def get_research_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific research session."""
        return self.research_sessions.get(session_id)
    
    def list_research_sessions(self) -> List[str]:
        """List all research session IDs."""
        return list(self.research_sessions.keys())
    
    def get_phase_results(self, phase: ResearchPhase) -> Optional[Dict[str, Any]]:
        """Get results from a specific research phase."""
        return self.phase_results.get(phase.value)
    
    def get_meeting_statistics(self) -> Dict[str, Any]:
        """Get statistics about meetings conducted."""
        stats = {
            'total_meetings': len(self.meeting_history),
            'successful_meetings': len([m for m in self.meeting_history if m.success]),
            'meeting_types': {},
            'phases_covered': {},
            'average_duration': 0,
            'total_participants': set()
        }
        
        if self.meeting_history:
            # Calculate meeting type distribution
            for meeting in self.meeting_history:
                meeting_type = meeting.meeting_type.value
                stats['meeting_types'][meeting_type] = stats['meeting_types'].get(meeting_type, 0) + 1
                
                phase = meeting.phase.value
                stats['phases_covered'][phase] = stats['phases_covered'].get(phase, 0) + 1
                
                stats['total_participants'].update(meeting.participants)
            
            # Calculate average duration
            durations = [m.end_time - m.start_time for m in self.meeting_history]
            stats['average_duration'] = sum(durations) / len(durations)
            stats['total_participants'] = len(stats['total_participants'])
        
        return stats