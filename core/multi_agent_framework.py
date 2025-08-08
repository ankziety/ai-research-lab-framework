"""
Multi-Agent AI-Powered Research Framework

A comprehensive multi-agent system that coordinates AI researchers to collaborate
on interdisciplinary research problems across any domain. Now enhanced with 
Virtual Lab methodology for structured meeting-based research collaboration.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import multi-agent components
from agents import (
    PrincipalInvestigatorAgent, AgentMarketplace, ScientificCriticAgent
)
from agents.base_agent import BaseAgent

# Import memory management components  
from memory import VectorDatabase, ContextManager, KnowledgeRepository

# Import Virtual Lab meeting system
from .virtual_lab import VirtualLabMeetingSystem

# Import cost management
try:
    from data.cost_manager import CostManager
    from data.manuscript_drafter import draft as draft_manuscript
    from data.literature_retriever import LiteratureRetriever
    from data.critic import Critic
    from data.results_visualizer import visualize as visualize_results
except ImportError:
    # Fallback for when running as module
    from ..data.cost_manager import CostManager
    from ..data.manuscript_drafter import draft as draft_manuscript
    from ..data.literature_retriever import LiteratureRetriever
    from ..data.critic import Critic
    from ..data.results_visualizer import visualize as visualize_results
from experiments.experiment import ExperimentRunner

logger = logging.getLogger(__name__)


def make_json_serializable(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable format."""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # Handle objects that don't have to_dict but have __dict__
        return {key: make_json_serializable(value) for key, value in obj.__dict__.items()}
    elif hasattr(obj, '__class__') and obj.__class__.__name__ in ['GeneralExpertAgent', 'BaseAgent', 'SimpleAgent']:
        # Handle agent objects that might not have to_dict
        return {
            'agent_id': getattr(obj, 'agent_id', 'unknown'),
            'role': getattr(obj, 'role', 'unknown'),
            'expertise': getattr(obj, 'expertise', []),
            'agent_type': obj.__class__.__name__
        }
    elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'MeetingRecord':
        # Handle MeetingRecord objects specifically
        return {
            'meeting_id': getattr(obj, 'meeting_id', 'unknown'),
            'meeting_type': getattr(obj, 'meeting_type', 'unknown'),
            'phase': getattr(obj, 'phase', 'unknown'),
            'participants': getattr(obj, 'participants', []),
            'agenda': make_json_serializable(getattr(obj, 'agenda', {})),
            'discussion_transcript': getattr(obj, 'discussion_transcript', []),
            'outcomes': getattr(obj, 'outcomes', {}),
            'decisions': getattr(obj, 'decisions', []),
            'action_items': getattr(obj, 'action_items', []),
            'start_time': getattr(obj, 'start_time', 0.0),
            'end_time': getattr(obj, 'end_time', 0.0),
            'success': getattr(obj, 'success', False)
        }
    elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'MeetingAgenda':
        # Handle MeetingAgenda objects specifically
        return {
            'meeting_id': getattr(obj, 'meeting_id', 'unknown'),
            'meeting_type': getattr(obj, 'meeting_type', 'unknown'),
            'phase': getattr(obj, 'phase', 'unknown'),
            'objectives': getattr(obj, 'objectives', []),
            'participants': getattr(obj, 'participants', []),
            'discussion_topics': getattr(obj, 'discussion_topics', []),
            'expected_outcomes': getattr(obj, 'expected_outcomes', []),
            'duration_minutes': getattr(obj, 'duration_minutes', 10)
        }
    elif hasattr(obj, 'value'):
        # Handle Enum objects
        return obj.value
    else:
        return obj


class MultiAgentResearchFramework:
    """
    Multi-agent AI-powered research framework that coordinates teams of AI experts
    to collaborate on research problems across any domain. Enhanced with Virtual Lab
    methodology for structured meeting-based research collaboration.
    
    The framework consists of:
    - Principal Investigator (PI) Agent that coordinates research
    - Agent Marketplace with domain expert agents  
    - Scientific Critic Agent for quality control
    - Virtual Lab Meeting System for structured collaboration
    - Vector Database for memory and context management
    - Knowledge Repository for validated findings
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the multi-agent research framework.
        
        Args:
            config: Configuration dictionary
        """
        self.config = self._load_config(config)
        
        # Initialize memory management
        self._init_memory_systems()
        
        # Initialize multi-agent system
        self._init_agent_system()
        
        # Initialize Virtual Lab meeting system
        self._init_virtual_lab()
        
        # Initialize original components for compatibility
        self._init_legacy_components()
        
        # Setup directories
        self._setup_directories()
        
        logger.info("Multi-Agent Research Framework with Virtual Lab initialized successfully")
    
    def _load_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load configuration with defaults."""
        # Handle None config
        if config is None:
            config = {}
            
        default_config = {
            # Memory configuration
            'vector_db_path': 'memory/vector_memory.db',
            'embedding_model': 'all-MiniLM-L6-v2',
            'max_context_length': 100000,
            
            # LLM API configuration for AI agents
            'openai_api_key': None,
            'anthropic_api_key': None,
            'default_llm_provider': 'openai',  # 'openai', 'anthropic', or 'local'
            'default_model': 'gpt-4o',
            
            # Agent configuration
            'max_agents_per_research': config.get('max_agents_per_research', 8),
            'agent_timeout': config.get('agent_timeout', 1800),  # 30 minutes default
            'agent_memory_limit': config.get('agent_memory_limit', 1000),  # Configurable memory
            'max_context_length': config.get('max_context_length', 10000),  # Configurable context
            
            # Legacy component configuration
            'experiment_db_path': 'experiments/experiments.db',
            'output_dir': 'output',
            'manuscript_dir': 'manuscripts',
            'visualization_dir': 'visualizations',
            'literature_api_url': None,
            'literature_api_key': None,
            'max_literature_results': 10,
            
            # Framework behavior
            'auto_critique': True,
            'auto_visualize': True,
            'store_all_interactions': True,
            'enable_memory_management': True
        }
        
        if config:
            default_config.update(config)
        
        return default_config
    
    def _init_memory_systems(self):
        """Initialize vector database, context manager, and knowledge repository."""
        logger.info("Initializing memory management systems...")
        
        # Initialize vector database
        self.vector_db = VectorDatabase(
            db_path=self.config['vector_db_path'],
            embedding_model=self.config['embedding_model']
        )
        
        # Initialize context manager
        self.context_manager = ContextManager(
            vector_db=self.vector_db,
            max_context_length=self.config['max_context_length']
        )
        
        # Initialize knowledge repository
        self.knowledge_repository = KnowledgeRepository(
            vector_db=self.vector_db
        )
        
        logger.info("Memory management systems initialized")
    
    def _init_agent_system(self):
        """Initialize the multi-agent system."""
        logger.info("Initializing multi-agent system...")
        
        # Prepare LLM configuration for agents
        llm_config = {
            'openai_api_key': self.config.get('openai_api_key'),
            'anthropic_api_key': self.config.get('anthropic_api_key'),
            'default_llm_provider': self.config.get('default_llm_provider', 'openai'),
            'default_model': self.config.get('default_model', 'gpt-4o')
        }
        
        # Initialize cost manager
        self.cost_config = {
            'budget_limit': self.config.get('budget_limit', 20.0),
            'cost_optimization': self.config.get('cost_optimization', True),
            'enable_dynamic_tools': self.config.get('enable_dynamic_tools', True),
            'default_model': self.config.get('default_model', 'gpt-4o-mini'),
            'premium_model': self.config.get('premium_model', 'gpt-4o'),
            'web_search_apis': self.config.get('web_search_apis', {}),
            'code_execution': self.config.get('code_execution', {
                'sandbox_enabled': True,
                'timeout_seconds': 30,
                'memory_limit_mb': 512
            })
        }
        
        self.cost_manager = CostManager(
            budget_limit=self.cost_config['budget_limit'],
            config=self.cost_config
        )
        
        # Initialize agent marketplace with LLM configuration and cost manager
        self.agent_marketplace = AgentMarketplace(
            llm_config=llm_config,
            cost_manager=self.cost_manager
        )
        
        # Initialize Principal Investigator with LLM configuration and cost manager
        pi_model_config = self.config.get('pi_model_config', {})
        pi_model_config.update(llm_config)
        self.pi_agent = PrincipalInvestigatorAgent(
            agent_id="PI_main",
            model_config=pi_model_config,
            cost_manager=self.cost_manager
        )
        
        # Initialize Scientific Critic with LLM configuration and cost manager
        critic_model_config = self.config.get('critic_model_config', {})
        critic_model_config.update(llm_config)
        self.scientific_critic = ScientificCriticAgent(
            agent_id="critic_main",
            model_config=critic_model_config,
            cost_manager=self.cost_manager
        )
        
        logger.info("Multi-agent system initialized")
    
    def _init_virtual_lab(self):
        """Initialize the Virtual Lab meeting system with cost management."""
        logger.info("Initializing Virtual Lab meeting system...")
        
        # Initialize Virtual Lab with the multi-agent components and cost management
        self.virtual_lab = VirtualLabMeetingSystem(
            pi_agent=self.pi_agent,
            scientific_critic=self.scientific_critic,
            agent_marketplace=self.agent_marketplace,
            config=self.cost_config,
            cost_manager=self.cost_manager
        )
        
        logger.info(f"Virtual Lab meeting system initialized with budget: ${self.config.get('budget_limit', 100.0):.2f}")
    
    def _init_legacy_components(self):
        logger.info("Initializing legacy components...")
        
        # Initialize experiment runner
        self.experiment_runner = ExperimentRunner(
            db_path=self.config.get('experiment_db_path')
        )
        
        # Initialize literature retriever with proper API key configuration
        literature_config = {
            'api_base_url': self.config.get('literature_api_url'),
            'max_results': self.config.get('max_literature_results', 10),
            'openai_api_key': self.config.get('openai_api_key'),
            'google_search_api_key': self.config.get('google_search_api_key'),
            'google_search_engine_id': self.config.get('google_search_engine_id'),
            'serpapi_key': self.config.get('serpapi_key'),
            'semantic_scholar_api_key': self.config.get('semantic_scholar_api_key'),
            'openalex_email': self.config.get('openalex_email'),
            'core_api_key': self.config.get('core_api_key')
        }
        
        self.literature_retriever = LiteratureRetriever(
            api_key=self.config.get('literature_api_key'),
            config=literature_config
        )
        
        # Initialize traditional critic
        self.traditional_critic = Critic()
        
        logger.info("Legacy components initialized")
    
    def _setup_directories(self):
        """Create necessary output directories."""
        dirs_to_create = [
            self.config['output_dir'],
            self.config['manuscript_dir'],
            self.config['visualization_dir'],
            'memory',
            'sessions'
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def conduct_research(self, research_question: str, 
                        constraints: Optional[Dict[str, Any]] = None,
                        context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Conduct research using the multi-agent framework.
        
        Args:
            research_question: The research question to investigate
            constraints: Optional constraints (budget, time, etc.)
            context: Optional additional context
            
        Returns:
            Complete research session results
        """
        logger.info(f"Starting multi-agent research session: {research_question[:100]}...")
        
        session_id = f"session_{int(time.time())}"
        
        # Store initial research question in context
        if self.config['store_all_interactions']:
            self.context_manager.add_to_context(
                session_id=session_id,
                content=f"Research Question: {research_question}",
                content_type="research_question",
                agent_id="human_researcher",
                importance_score=1.0,
                metadata={'constraints': constraints, 'context': context}
            )
        
        try:
            # Step 1: PI analyzes and coordinates research
            coordination_result = self.pi_agent.coordinate_research_session(
                problem_description=research_question,
                marketplace=self.agent_marketplace,
                constraints=constraints,
                session_id=session_id
            )
            
            # Store coordination results in context
            if self.config['store_all_interactions']:
                self.context_manager.add_to_context(
                    session_id=session_id,
                    content=f"PI Analysis: {coordination_result.get('analysis', {})}",
                    content_type="pi_analysis",
                    agent_id="PI_main",
                    importance_score=0.9
                )
            
            # Step 2: Execute multi-agent collaboration
            collaboration_results = self._execute_agent_collaboration(
                session_id=session_id,
                coordination_result=coordination_result
            )
            
            # Step 3: Scientific critique of results
            critique_result = self._perform_scientific_critique(
                session_id=session_id,
                research_outputs=collaboration_results
            )
            
            # Step 4: Synthesize final results
            final_synthesis = self._synthesize_research_results(
                session_id=session_id,
                coordination_result=coordination_result,
                collaboration_results=collaboration_results,
                critique_result=critique_result
            )
            
            # Step 5: Extract and store validated findings
            validated_findings = self._extract_validated_findings(
                session_id=session_id,
                synthesis=final_synthesis,
                critique_result=critique_result
            )
            
            # Compile complete research session
            research_session = {
                'session_id': session_id,
                'research_question': research_question,
                'start_time': coordination_result.get('start_time'),
                'end_time': time.time(),
                'constraints': constraints,
                'context': context,
                'coordination_result': coordination_result,
                'collaboration_results': collaboration_results,
                'critique_result': critique_result,
                'synthesis': final_synthesis,
                'validated_findings': validated_findings,
                'status': 'completed'
            }
            
            logger.info(f"Multi-agent research session completed: {session_id}")
            return research_session
            
        except Exception as e:
            logger.error(f"Research session {session_id} failed: {e}")
            return {
                'session_id': session_id,
                'research_question': research_question,
                'status': 'failed',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def conduct_virtual_lab_research(self, research_question: str, 
                                   constraints: Optional[Dict[str, Any]] = None,
                                   context: Optional[Dict[str, Any]] = None,
                                   session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Conduct research using the Virtual Lab methodology with structured meetings.
        
        This is the enhanced research method that uses the Virtual Lab approach
        with meeting-based collaboration between AI agents.
        
        Args:
            research_question: The research question to investigate
            constraints: Optional constraints (budget, time, etc.)
            context: Optional additional context
            session_id: Optional session ID to use (if not provided, will generate one)
            
        Returns:
            Complete Virtual Lab research session results
        """
        logger.info(f"Starting Virtual Lab research: {research_question[:100]}...")
        
        try:
            # Use Virtual Lab meeting system for structured research
            vlab_results = self.virtual_lab.conduct_research_session(
                research_question=research_question,
                constraints=constraints,
                context=context,
                session_id=session_id
            )
            
            # Store results in context manager if enabled
            if self.config['store_all_interactions'] and vlab_results.get('session_id'):
                self.context_manager.add_to_context(
                    session_id=vlab_results['session_id'],
                    content=f"Virtual Lab Research Results: {vlab_results.get('final_results', {})}",
                    content_type="vlab_research_results",
                    agent_id="virtual_lab_system",
                    importance_score=1.0,
                    metadata=make_json_serializable(vlab_results)
                )
            
            logger.info(f"Virtual Lab research completed: {vlab_results.get('session_id', 'unknown')}")
            return vlab_results
            
        except Exception as e:
            logger.error(f"Virtual Lab research failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'research_question': research_question,
                'timestamp': time.time()
            }
    
    def _execute_agent_collaboration(self, session_id: str, 
                                   coordination_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute collaboration between hired agents."""
        logger.info(f"Executing agent collaboration for session {session_id}")
        
        hired_agents = coordination_result.get('hired_agents', {}).get('hired_agents', {})
        task_assignments = coordination_result.get('task_assignments', {})
        
        collaboration_results = {
            'agent_outputs': {},
            'agent_interactions': [],
            'cross_pollination': {}
        }
        
        # Execute tasks by agents
        for task_id, assignment in task_assignments.items():
            agent_id = assignment['agent_id']
            task = assignment['task']
            
            # Find the hired agent (now stored as dictionary)
            agent_dict = None
            for expertise, agent_data in hired_agents.items():
                if agent_data.get('agent_id') == agent_id:
                    agent_dict = agent_data
                    break
            
            if not agent_dict:
                logger.warning(f"Agent {agent_id} not found for task {task_id}")
                continue
            
            # Create a simple agent object for execution
            agent = self._create_simple_agent_from_dict(agent_dict)
            
            # Get relevant context for the agent
            if self.config['enable_memory_management']:
                enhanced_task_description = self.context_manager.inject_relevant_context(
                    session_id=session_id,
                    current_prompt=task['description']
                )
            else:
                enhanced_task_description = task['description']
            
            # Execute task
            try:
                agent_response = agent.receive_message(
                    sender_id="PI_main",
                    message=enhanced_task_description,
                    context={'task_id': task_id, 'session_id': session_id}
                )
                
                collaboration_results['agent_outputs'][agent_id] = {
                    'task_id': task_id,
                    'response': agent_response,
                    'timestamp': time.time(),
                    'expertise': task['expertise_required']
                }
                
                # Store agent output in context
                if self.config['store_all_interactions']:
                    self.context_manager.add_to_context(
                        session_id=session_id,
                        content=agent_response,
                        content_type="agent_output",
                        agent_id=agent_id,
                        importance_score=0.8,
                        metadata={'task_id': task_id, 'expertise': task['expertise_required']}
                    )
                
                logger.debug(f"Agent {agent_id} completed task {task_id}")
                
            except Exception as e:
                logger.error(f"Agent {agent_id} failed task {task_id}: {e}")
                collaboration_results['agent_outputs'][agent_id] = {
                    'task_id': task_id,
                    'error': str(e),
                    'timestamp': time.time()
                }
        
        # Facilitate cross-agent interactions
        collaboration_results['cross_pollination'] = self._facilitate_cross_agent_interaction(
            session_id=session_id,
            hired_agents=hired_agents,
            agent_outputs=collaboration_results['agent_outputs']
        )
        
        return collaboration_results
    
    def _create_simple_agent_from_dict(self, agent_dict: Dict[str, Any]) -> Any:
        """Create a simple agent object from dictionary for execution."""
        from agents.base_agent import BaseAgent
        
        class SimpleAgent(BaseAgent):
            def __init__(self, agent_dict):
                super().__init__(
                    agent_id=agent_dict['agent_id'],
                    role=agent_dict['role'],
                    expertise=agent_dict['expertise']
                )
            
            def generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
                """Generate a simple response based on role and expertise."""
                return f"Agent {self.agent_id} ({self.role}) response: {prompt[:100]}..."
        
        return SimpleAgent(agent_dict)
    
    def _facilitate_cross_agent_interaction(self, session_id: str,
                                          hired_agents: Dict[str, Any],
                                          agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Facilitate interaction between agents for cross-pollination of ideas."""
        logger.debug(f"Facilitating cross-agent interaction for session {session_id}")
        
        cross_pollination = {
            'interactions': [],
            'synthesis_points': []
        }
        
        # Create summary of all agent outputs
        output_summary = []
        for agent_id, output_data in agent_outputs.items():
            if 'response' in output_data:
                output_summary.append(f"{agent_id}: {output_data['response'][:200]}...")
        
        if len(output_summary) < 2:
            return cross_pollination
        
        # Have each agent comment on others' findings
        try:
            # Check if hired_agents is actually a dictionary
            if isinstance(hired_agents, dict):
                for expertise, agent_dict in hired_agents.items():
                    if not isinstance(agent_dict, dict):
                        logger.warning(f"Invalid agent_dict type for {expertise}: {type(agent_dict)}")
                        continue
                    
                    try:
                        # Create a simple agent object for interaction
                        agent = self._create_simple_agent_from_dict(agent_dict)
                        
                        # Filter out this agent's own results
                        other_results = [summary for summary in output_summary 
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
                            
                            # For SimpleAgent objects, generate a simple response
                            if hasattr(agent, 'generate_response'):
                                cross_response = agent.generate_response(cross_prompt, {
                                    'session_id': session_id,
                                    'interaction_type': 'cross_pollination',
                                    'other_results': other_results
                                })
                            else:
                                cross_response = f"Agent {agent.agent_id} ({agent.role}) cross-analysis: {expertise} insights on findings"
                            
                            cross_pollination['interactions'].append({
                                'agent_id': agent.agent_id,
                                'expertise': expertise,
                                'cross_analysis': cross_response,
                                'timestamp': time.time()
                            })
                            
                    except Exception as e:
                        logger.warning(f"Cross-pollination failed for agent {agent_dict.get('agent_id', 'unknown')}: {e}")
            else:
                logger.warning(f"hired_agents is not a dictionary: {type(hired_agents)}")
        except Exception as e:
            logger.error(f"Error processing hired agents in cross-pollination: {e}")
        
        return cross_pollination
    
    def _perform_scientific_critique(self, session_id: str,
                                   research_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Perform scientific critique of research outputs."""
        logger.info(f"Performing scientific critique for session {session_id}")
        
        # Compile all research outputs for critique
        all_outputs = []
        
        # Add agent outputs
        for agent_id, output_data in research_outputs.get('agent_outputs', {}).items():
            if 'response' in output_data:
                all_outputs.append(output_data['response'])
        
        # Add cross-pollination insights
        for interaction in research_outputs.get('cross_pollination', {}).get('interactions', []):
            all_outputs.append(interaction['cross_analysis'])
        
        combined_output = "\n\n".join(all_outputs)
        
        # Perform critique using scientific critic agent
        critique_result = self.scientific_critic.critique_research_output(
            output_content=combined_output,
            output_type="multi_agent_research",
            context={'session_id': session_id}
        )
        
        # Store critique in context
        if self.config['store_all_interactions']:
            self.context_manager.add_to_context(
                session_id=session_id,
                content=f"Scientific Critique: {critique_result}",
                content_type="scientific_critique",
                agent_id="critic_main",
                importance_score=0.9,
                metadata=critique_result
            )
        
        return critique_result
    
    def _synthesize_research_results(self, session_id: str,
                                   coordination_result: Dict[str, Any],
                                   collaboration_results: Dict[str, Any],
                                   critique_result: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize all research results into coherent findings."""
        logger.info(f"Synthesizing research results for session {session_id}")
        
        # Use PI agent to synthesize results
        synthesis_prompt = f"""
        Please synthesize the following research session results:
        
        Research Question: {coordination_result.get('problem_description', 'Unknown')}
        
        Agent Findings:
        {self._format_agent_findings(collaboration_results.get('agent_outputs', {}))}
        
        Cross-Agent Insights:
        {self._format_cross_pollination(collaboration_results.get('cross_pollination', {}))}
        
        Scientific Critique Summary:
        Overall Score: {critique_result.get('overall_score', 'N/A')}/100
        Key Issues: {', '.join(critique_result.get('critical_issues', []))}
        Recommendations: {', '.join(critique_result.get('recommendations', []))}
        
        Please provide:
        1. Key research findings
        2. Confidence assessment
        3. Methodological strengths and limitations
        4. Recommendations for future research
        """
        
        synthesis_response = self.pi_agent.receive_message(
            sender_id="framework",
            message=synthesis_prompt,
            context={'session_id': session_id, 'task_type': 'synthesis'}
        )
        
        synthesis = {
            'synthesis_text': synthesis_response,
            'confidence_score': self._extract_confidence_score(synthesis_response),
            'key_findings': self._extract_key_findings(synthesis_response),
            'recommendations': self._extract_recommendations(synthesis_response),
            'timestamp': time.time()
        }
        
        # Store synthesis in context
        if self.config['store_all_interactions']:
            self.context_manager.add_to_context(
                session_id=session_id,
                content=synthesis_response,
                content_type="research_synthesis",
                agent_id="PI_main",
                importance_score=1.0,
                metadata=synthesis
            )
        
        return synthesis
    
    def _extract_validated_findings(self, session_id: str, 
                                  synthesis: Dict[str, Any],
                                  critique_result: Dict[str, Any]) -> List[str]:
        """Extract and store validated findings."""
        validated_findings = []
        
        # Only store findings if critique score is high enough
        if critique_result.get('overall_score', 0) >= 70:
            key_findings = synthesis.get('key_findings', [])
            confidence_score = synthesis.get('confidence_score', 0.5)
            
            for finding in key_findings:
                finding_id = self.knowledge_repository.add_validated_finding(
                    finding_text=finding,
                    research_domain="multi_agent_research",
                    confidence_score=confidence_score,
                    evidence_sources=[session_id],
                    validating_agents=["PI_main", "critic_main"],
                    session_id=session_id
                )
                validated_findings.append(finding_id)
        
        return validated_findings
    
    def _format_agent_findings(self, agent_outputs: Dict[str, Any]) -> str:
        """Format agent findings for synthesis."""
        formatted = []
        for agent_id, output_data in agent_outputs.items():
            if 'response' in output_data:
                formatted.append(f"{agent_id}: {output_data['response']}")
        return "\n\n".join(formatted)
    
    def _format_cross_pollination(self, cross_pollination: Dict[str, Any]) -> str:
        """Format cross-pollination insights."""
        formatted = []
        for interaction in cross_pollination.get('interactions', []):
            formatted.append(f"{interaction['agent_id']}: {interaction['cross_analysis']}")
        return "\n\n".join(formatted)
    
    def _extract_confidence_score(self, text: str) -> float:
        """Extract confidence score from synthesis text."""
        # Simplified extraction - real implementation would use NLP
        confidence_keywords = {
            'very high': 0.9, 'high': 0.8, 'moderate': 0.6, 
            'low': 0.4, 'very low': 0.2
        }
        
        text_lower = text.lower()
        for keyword, score in confidence_keywords.items():
            if keyword in text_lower:
                return score
        
        return 0.7  # Default moderate confidence
    
    def _extract_key_findings(self, text: str) -> List[str]:
        """Extract key findings from synthesis text."""
        # Simplified extraction
        lines = text.split('\n')
        findings = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('- ') or line.startswith('* '):
                findings.append(line[2:])
            elif 'finding' in line.lower() and len(line) > 20:
                findings.append(line)
        
        return findings[:5]  # Limit to top 5 findings
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from synthesis text."""
        # Simplified extraction
        lines = text.split('\n')
        recommendations = []
        
        in_recommendations_section = False
        for line in lines:
            line = line.strip()
            if 'recommendation' in line.lower():
                in_recommendations_section = True
                continue
            
            if in_recommendations_section:
                if line.startswith('- ') or line.startswith('* '):
                    recommendations.append(line[2:])
                elif line and not line.startswith('-') and len(recommendations) > 0:
                    break  # End of recommendations section
        
        return recommendations[:3]  # Limit to top 3 recommendations
    
    # Compatibility methods for legacy functionality
    def run_experiment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run experiment using legacy experiment runner."""
        return self.experiment_runner.run_experiment(params)
    
    def retrieve_literature(self, query: str, max_results: Optional[int] = None) -> List[Dict]:
        """Retrieve literature using legacy literature retriever."""
        if max_results is None:
            max_results = self.config.get('max_literature_results', 10)
        return self.literature_retriever.search(query, max_results)
    
    def draft_manuscript(self, results: List[Dict[str, Any]], 
                        context: Dict[str, Any]) -> str:
        """Draft manuscript using legacy manuscript drafter."""
        return draft_manuscript(results, context)
    
    def visualize_results(self, results: List[Dict], out_path: str) -> None:
        """Generate visualizations using legacy visualizer."""
        visualize_results(results, out_path)
    
    # Memory and knowledge management methods
    def search_knowledge(self, query: str, research_domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search validated knowledge base."""
        return self.knowledge_repository.search_findings(query, research_domain)
    
    def get_agent_performance(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent performance summary."""
        return self.knowledge_repository.get_agent_performance_summary(agent_id)
    
    def get_framework_statistics(self) -> Dict[str, Any]:
        """Get comprehensive framework statistics."""
        stats = {
            'agent_marketplace': self.agent_marketplace.get_marketplace_statistics(),
            'knowledge_repository': self.knowledge_repository.get_repository_stats(),
            'vector_database': self.vector_db.get_stats(),
            'context_manager': self.context_manager.get_context_stats()
        }
        
        # Add Virtual Lab statistics
        if hasattr(self, 'virtual_lab'):
            stats['virtual_lab'] = self.virtual_lab.get_meeting_statistics()
        
        return stats
    
    # Virtual Lab specific methods
    
    def get_virtual_lab_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific Virtual Lab research session."""
        if hasattr(self, 'virtual_lab'):
            return self.virtual_lab.get_research_session(session_id)
        return None
    
    def list_virtual_lab_sessions(self) -> List[str]:
        """List all Virtual Lab research session IDs."""
        if hasattr(self, 'virtual_lab'):
            return self.virtual_lab.list_research_sessions()
        return []
    
    def get_meeting_history(self, limit: Optional[int] = None) -> List[Any]:
        """Get Virtual Lab meeting history."""
        if hasattr(self, 'virtual_lab'):
            return self.virtual_lab.get_meeting_history(limit)
        return []
    
    def get_virtual_lab_statistics(self) -> Dict[str, Any]:
        """Get detailed Virtual Lab meeting statistics."""
        if hasattr(self, 'virtual_lab'):
            return self.virtual_lab.get_meeting_statistics()
        return {}
    
    def cleanup_old_data(self, days_old: int = 30):
        """Clean up old data from memory systems."""
        self.vector_db.cleanup_old_content(days_old)
        self.context_manager.cleanup_old_sessions(days_old * 24)  # Convert to hours
    
    def close(self):
        """Close all database connections and cleanup resources."""
        if hasattr(self, 'vector_db'):
            self.vector_db.close()
        if hasattr(self, 'experiment_runner'):
            self.experiment_runner.close()
        logger.info("Multi-Agent Research Framework closed")

    def get_active_agents(self) -> List[BaseAgent]:
        """Get list of currently active agents."""
        active_agents = []
        
        # Get agents from marketplace
        if hasattr(self, 'agent_marketplace') and self.agent_marketplace:
            active_agents.extend(self.agent_marketplace.hired_agents.values())
        
        # Get PI agent if active
        if hasattr(self, 'pi_agent') and self.pi_agent:
            active_agents.append(self.pi_agent)
        
        # Get scientific critic if active
        if hasattr(self, 'scientific_critic') and self.scientific_critic:
            active_agents.append(self.scientific_critic)
        
        return active_agents

    def get_agent_activity_log(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get agent activity log for monitoring."""
        activity_log = []
        
        # Get activity from virtual lab if available
        if hasattr(self, 'virtual_lab') and self.virtual_lab:
            activity_log = self.virtual_lab.get_agent_activity_log(session_id)
        else:
            # Fallback to in-memory storage
            if hasattr(self, '_agent_activity_log'):
                if session_id:
                    activity_log = [activity for activity in self._agent_activity_log if activity.get('session_id') == session_id]
                else:
                    activity_log = self._agent_activity_log.copy()
        
        return activity_log

    def log_agent_activity(self, agent_id: str, activity_type: str, message: str, 
                          session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Log agent activity for web UI monitoring."""
        activity = {
            'agent_id': agent_id,
            'activity_type': activity_type,
            'message': message,
            'timestamp': time.time(),
            'session_id': session_id,
            'metadata': metadata or {}
        }
        
        # Store in memory for current session
        if not hasattr(self, '_agent_activity_log'):
            self._agent_activity_log = []
        
        self._agent_activity_log.append(activity)
        
        # Keep only recent activities (last 1000)
        if len(self._agent_activity_log) > 1000:
            self._agent_activity_log = self._agent_activity_log[-1000:]
        
        logger.info(f"Agent activity: {agent_id} - {activity_type}: {message}")
        
        return activity

    def get_chat_logs(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get chat logs including agent thoughts and communications."""
        chat_logs = []
        
        # Get logs from virtual lab if available
        if hasattr(self, 'virtual_lab') and self.virtual_lab:
            chat_logs = self.virtual_lab.get_chat_logs(session_id)
        else:
            # Fallback to in-memory storage
            if hasattr(self, '_chat_logs'):
                if session_id:
                    chat_logs = [log for log in self._chat_logs if log.get('session_id') == session_id]
                else:
                    chat_logs = self._chat_logs.copy()
        
        return chat_logs

    def log_chat_message(self, log_type: str, author: str, message: str, 
                        session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Log chat messages for web UI monitoring."""
        chat_log = {
            'log_type': log_type,
            'author': author,
            'message': message,
            'timestamp': time.time(),
            'session_id': session_id,
            'metadata': metadata or {}
        }
        
        # Store in memory for current session
        if not hasattr(self, '_chat_logs'):
            self._chat_logs = []
        
        self._chat_logs.append(chat_log)
        
        # Keep only recent logs (last 2000)
        if len(self._chat_logs) > 2000:
            self._chat_logs = self._chat_logs[-2000:]
        
        logger.info(f"Chat log: {log_type} - {author}: {message[:100]}...")
        
        return chat_log

    def calculate_text_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate text analysis metrics for monitoring."""
        if not text:
            return {'word_count': 0, 'sentence_count': 0, 'avg_sentence_length': 0}
        
        # Basic text analysis
        words = text.split()
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': round(avg_sentence_length, 2)
        }


def create_framework(config: Optional[Dict[str, Any]] = None) -> MultiAgentResearchFramework:
    """
    Factory function to create a Multi-Agent Research Framework instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured MultiAgentResearchFramework instance
    """
    return MultiAgentResearchFramework(config)


# Backward compatibility alias
AIPoweredResearchFramework = MultiAgentResearchFramework