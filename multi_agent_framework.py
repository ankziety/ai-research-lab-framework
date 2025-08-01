"""
Multi-Agent AI-Powered Research Framework

A comprehensive multi-agent system that coordinates AI researchers to collaborate
on interdisciplinary research problems across any domain.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import multi-agent components
from agents import (
    PrincipalInvestigatorAgent, AgentMarketplace, ScientificCriticAgent
)

# Import memory management components  
from memory import VectorDatabase, ContextManager, KnowledgeRepository

# Import original framework components for backward compatibility
from manuscript_drafter import draft as draft_manuscript
from literature_retriever import LiteratureRetriever
from critic import Critic
from results_visualizer import visualize as visualize_results
from experiments.experiment import ExperimentRunner

logger = logging.getLogger(__name__)


class MultiAgentResearchFramework:
    """
    Multi-agent AI-powered research framework that coordinates teams of AI experts
    to collaborate on research problems across any domain.
    
    The framework consists of:
    - Principal Investigator (PI) Agent that coordinates research
    - Agent Marketplace with domain expert agents  
    - Scientific Critic Agent for quality control
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
        
        # Initialize original components for compatibility
        self._init_legacy_components()
        
        # Setup directories
        self._setup_directories()
        
        logger.info("Multi-Agent Research Framework initialized successfully")
    
    def _load_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load configuration with defaults."""
        default_config = {
            # Memory configuration
            'vector_db_path': 'memory/vector_memory.db',
            'embedding_model': 'all-MiniLM-L6-v2',
            'max_context_length': 4000,
            
            # LLM API configuration for AI agents
            'openai_api_key': None,
            'anthropic_api_key': None,
            'default_llm_provider': 'openai',  # 'openai', 'anthropic', or 'local'
            'default_model': 'gpt-4',
            
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
            'default_model': self.config.get('default_model', 'gpt-4')
        }
        
        # Initialize agent marketplace with LLM configuration
        self.agent_marketplace = AgentMarketplace(llm_config=llm_config)
        
        # Initialize Principal Investigator with LLM configuration
        pi_model_config = self.config.get('pi_model_config', {})
        pi_model_config.update(llm_config)
        self.pi_agent = PrincipalInvestigatorAgent(
            agent_id="PI_main",
            model_config=pi_model_config
        )
        
        # Initialize Scientific Critic with LLM configuration
        critic_model_config = self.config.get('critic_model_config', {})
        critic_model_config.update(llm_config)
        self.scientific_critic = ScientificCriticAgent(
            agent_id="critic_main",
            model_config=critic_model_config
        )
        
        logger.info("Multi-agent system initialized")
    
    def _init_legacy_components(self):
        """Initialize legacy components for backward compatibility."""
        logger.info("Initializing legacy components...")
        
        # Initialize experiment runner
        self.experiment_runner = ExperimentRunner(
            db_path=self.config.get('experiment_db_path')
        )
        
        # Initialize literature retriever
        self.literature_retriever = LiteratureRetriever(
            api_base_url=self.config.get('literature_api_url'),
            api_key=self.config.get('literature_api_key')
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
        Conduct a complete multi-agent research session.
        
        Args:
            research_question: The research question to investigate
            constraints: Optional constraints (budget, time, etc.)
            context: Optional additional context
            
        Returns:
            Complete research session results
        """
        logger.info(f"Starting multi-agent research session: {research_question[:100]}...")
        
        session_id = f"research_{int(time.time())}"
        
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
                constraints=constraints
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
            
            # Find the hired agent
            agent = None
            for expertise, hired_agent in hired_agents.items():
                if hired_agent.agent_id == agent_id:
                    agent = hired_agent
                    break
            
            if not agent:
                logger.warning(f"Agent {agent_id} not found for task {task_id}")
                continue
            
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
        for expertise, agent in hired_agents.items():
            if agent.agent_id not in agent_outputs:
                continue
            
            # Create cross-pollination prompt
            other_findings = [summary for summary in output_summary 
                            if not summary.startswith(agent.agent_id)]
            
            if other_findings:
                cross_prompt = f"""
                Based on your expertise in {expertise}, please provide insights on these findings from other experts:
                
                {chr(10).join(other_findings)}
                
                What connections, contradictions, or synergies do you observe?
                """
                
                try:
                    cross_response = agent.receive_message(
                        sender_id="PI_main",
                        message=cross_prompt,
                        context={'session_id': session_id, 'interaction_type': 'cross_pollination'}
                    )
                    
                    cross_pollination['interactions'].append({
                        'agent_id': agent.agent_id,
                        'expertise': expertise,
                        'cross_analysis': cross_response,
                        'timestamp': time.time()
                    })
                    
                    # Store interaction in context
                    if self.config['store_all_interactions']:
                        self.context_manager.add_to_context(
                            session_id=session_id,
                            content=cross_response,
                            content_type="cross_pollination",
                            agent_id=agent.agent_id,
                            importance_score=0.7,
                            metadata={'interaction_type': 'cross_analysis'}
                        )
                    
                except Exception as e:
                    logger.warning(f"Cross-pollination failed for agent {agent.agent_id}: {e}")
        
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
        return {
            'agent_marketplace': self.agent_marketplace.get_marketplace_statistics(),
            'knowledge_repository': self.knowledge_repository.get_repository_stats(),
            'vector_database': self.vector_db.get_stats(),
            'context_manager': self.context_manager.get_context_stats()
        }
    
    def cleanup_old_data(self, days_old: int = 30):
        """Clean up old data from memory systems."""
        self.vector_db.cleanup_old_content(days_old)
        self.context_manager.cleanup_old_sessions(days_old * 24)  # Convert to hours
    
    def close(self):
        """Close all database connections and cleanup resources."""
        if hasattr(self, 'vector_db'):
            self.vector_db.close()
        logger.info("Multi-Agent Research Framework closed")


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