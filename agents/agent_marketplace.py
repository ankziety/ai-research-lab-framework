"""
Agent Marketplace - Manages pool of expert agents and hiring decisions.
"""

import logging
from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent
from .scientific_critic import ScientificCriticAgent

logger = logging.getLogger(__name__)


class GeneralExpertAgent(BaseAgent):
    """
    General expert agent that can be configured for any domain.
    Uses default implementations from BaseAgent.
    """
    
    def __init__(self, agent_id: str, role: str, expertise: List[str], 
                 model_config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, role, expertise, model_config)


class AgentMarketplace:
    """
    Marketplace that manages a pool of expert agents and handles hiring decisions.
    """
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        self.available_agents = {}
        self.hired_agents = {}
        self.agent_registry = {}
        self.hiring_history = []
        self.llm_config = llm_config or {}
        
        # Initialize default agents
        self._initialize_default_agents()
        
        logger.info("Agent Marketplace initialized")
    
    def _create_expert_agent(self, agent_id: str, role: str, expertise: List[str]):
        """Create a general expert agent with specified role and expertise."""
        return GeneralExpertAgent(agent_id, role, expertise, self.llm_config)
    
    def _initialize_default_agents(self):
        """Initialize a general set of versatile expert agents."""
        default_agents = [
            # General research agents 
            self._create_expert_agent("research_methodology_1", "Research Methodology Expert", 
                                    ["Research Design", "Statistical Analysis", "Study Methodology", "Data Collection"]),
            self._create_expert_agent("literature_researcher_1", "Literature Research Expert",
                                    ["Literature Review", "Citation Analysis", "Academic Writing", "Systematic Reviews"]),
            self._create_expert_agent("data_scientist_1", "Data Science Expert",
                                    ["Data Analysis", "Machine Learning", "Statistical Modeling", "Data Visualization"]),
            self._create_expert_agent("critical_analyst_1", "Critical Analysis Expert",
                                    ["Critical Thinking", "Bias Detection", "Logical Analysis", "Quality Assessment"]),
            # Add backup agents for high demand scenarios
            self._create_expert_agent("general_researcher_1", "General Research Expert",
                                    ["Interdisciplinary Research", "Problem Solving", "Knowledge Synthesis"]),
        ]
        
        for agent in default_agents:
            self.register_agent(agent)
            
        logger.info(f"Initialized {len(default_agents)} general-purpose default agents")
    
    def register_agent(self, agent: BaseAgent):
        """
        Register a new agent in the marketplace.
        
        Args:
            agent: Agent instance to register
        """
        if agent.agent_id in self.agent_registry:
            raise ValueError(f"Agent {agent.agent_id} already registered")
        
        self.agent_registry[agent.agent_id] = agent
        self.available_agents[agent.agent_id] = agent
        
        logger.info(f"Registered agent: {agent.agent_id} ({agent.role})")
    
    def unregister_agent(self, agent_id: str):
        """
        Unregister an agent from the marketplace.
        
        Args:
            agent_id: ID of agent to unregister
        """
        if agent_id not in self.agent_registry:
            raise ValueError(f"Agent {agent_id} not found in registry")
        
        # Remove from all tracking dictionaries
        self.agent_registry.pop(agent_id, None)
        self.available_agents.pop(agent_id, None)
        self.hired_agents.pop(agent_id, None)
        
        logger.info(f"Unregistered agent: {agent_id}")
    
    def get_agents_by_expertise(self, expertise_domain: str) -> List[BaseAgent]:
        """
        Get available agents with specific expertise.
        
        Args:
            expertise_domain: Domain of expertise to search for
            
        Returns:
            List of agents with matching expertise
        """
        matching_agents = []
        
        # Map expertise domains to agent types (more general)
        domain_mapping = {
            'research_methodology': ['Research Methodology Expert'],
            'literature': ['Literature Research Expert', 'Literature Researcher'],
            'data_science': ['Data Science Expert', 'Data Scientist'],
            'critical_analysis': ['Critical Analysis Expert', 'Scientific Critic'],
            'general_research': ['General Research Expert'],
            # Legacy mappings for backward compatibility
            'psychology': ['Psychology Expert'],
            'neuroscience': ['Neuroscience Expert'], 
            'ophthalmology': ['Ophthalmology Expert'],
            'critic': ['Scientific Critic', 'Critical Analysis Expert'],
            # Additional domain mappings for common research areas
            'biomedical_engineering': ['Biomedical Engineering Expert', 'General Research Expert'],
            'materials_science': ['Materials Science Expert', 'General Research Expert'],
            'signal_processing': ['Signal Processing Expert', 'Data Science Expert'],
            'clinical_research': ['Clinical Research Expert', 'General Research Expert'],
            'engineering': ['Engineering Expert', 'General Research Expert'],
            'biology': ['Biology Expert', 'General Research Expert'],
            'chemistry': ['Chemistry Expert', 'General Research Expert'],
            'physics': ['Physics Expert', 'General Research Expert'],
            'computer_science': ['Computer Science Expert', 'Data Science Expert'],
            'mathematics': ['Mathematics Expert', 'Data Science Expert']
        }
        
        target_roles = domain_mapping.get(expertise_domain, [])
        
        for agent in self.available_agents.values():
            if agent.role in target_roles:
                matching_agents.append(agent)
        
        # If no agents found, try to find agents with similar expertise in their expertise list
        if not matching_agents:
            for agent in self.available_agents.values():
                for expertise in agent.expertise:
                    if expertise_domain.lower() in expertise.lower() or expertise.lower() in expertise_domain.lower():
                        matching_agents.append(agent)
                        break
        
        logger.info(f"Found {len(matching_agents)} agents for expertise: {expertise_domain}")
        return matching_agents
    
    def get_agent_by_id(self, agent_id: str) -> Optional[BaseAgent]:
        """
        Get agent by ID from registry.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent instance or None if not found
        """
        return self.agent_registry.get(agent_id)
    
    def hire_agent(self, agent_id: str) -> bool:
        """
        Hire an agent (move from available to hired).
        
        Args:
            agent_id: ID of agent to hire
            
        Returns:
            True if successfully hired, False otherwise
        """
        if agent_id not in self.available_agents:
            logger.warning(f"Agent {agent_id} not available for hiring")
            return False
        
        agent = self.available_agents.pop(agent_id)
        self.hired_agents[agent_id] = agent
        
        # Record hiring transaction
        hiring_record = {
            'agent_id': agent_id,
            'agent_role': agent.role,
            'hire_timestamp': self._get_timestamp(),
            'performance_at_hire': agent.performance_metrics.copy()
        }
        self.hiring_history.append(hiring_record)
        
        logger.info(f"Hired agent: {agent_id}")
        return True
    
    def release_agent(self, agent_id: str) -> bool:
        """
        Release an agent (move from hired back to available).
        
        Args:
            agent_id: ID of agent to release
            
        Returns:
            True if successfully released, False otherwise
        """
        if agent_id not in self.hired_agents:
            logger.warning(f"Agent {agent_id} not currently hired")
            return False
        
        agent = self.hired_agents.pop(agent_id)
        self.available_agents[agent_id] = agent
        
        logger.info(f"Released agent: {agent_id}")
        return True
    
    def get_agent_performance_ranking(self, expertise_domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get agents ranked by performance metrics.
        
        Args:
            expertise_domain: Optional domain to filter by
            
        Returns:
            List of agent performance summaries, ranked by quality
        """
        agents_to_rank = []
        
        if expertise_domain:
            agents_to_rank = self.get_agents_by_expertise(expertise_domain)
        else:
            agents_to_rank = list(self.agent_registry.values())
        
        # Calculate composite performance score
        agent_scores = []
        for agent in agents_to_rank:
            metrics = agent.performance_metrics
            
            composite_score = (
                metrics['average_quality_score'] * 0.4 +
                metrics['success_rate'] * 0.3 +
                (min(1.0, metrics['tasks_completed'] / 10.0)) * 0.2 +  # Experience factor
                (0.1 if agent.agent_id in self.available_agents else 0.0) * 0.1  # Availability bonus
            )
            
            agent_scores.append({
                'agent_id': agent.agent_id,
                'role': agent.role,
                'expertise': agent.expertise,
                'composite_score': composite_score,
                'performance_metrics': metrics.copy(),
                'status': 'available' if agent.agent_id in self.available_agents else 'hired'
            })
        
        # Sort by composite score (highest first)
        agent_scores.sort(key=lambda x: x['composite_score'], reverse=True)
        
        return agent_scores
    
    def get_marketplace_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the marketplace.
        
        Returns:
            Dictionary containing marketplace statistics
        """
        total_agents = len(self.agent_registry)
        available_agents = len(self.available_agents)
        hired_agents = len(self.hired_agents)
        
        # Count agents by role
        role_counts = {}
        for agent in self.agent_registry.values():
            role_counts[agent.role] = role_counts.get(agent.role, 0) + 1
        
        # Calculate utilization metrics
        utilization_rate = hired_agents / max(1, total_agents)
        
        # Get hiring activity
        recent_hires = len([h for h in self.hiring_history if self._is_recent(h['hire_timestamp'])])
        
        return {
            'total_agents': total_agents,
            'available_agents': available_agents,
            'hired_agents': hired_agents,
            'utilization_rate': utilization_rate,
            'agents_by_role': role_counts,
            'total_hirings': len(self.hiring_history),
            'recent_hires_24h': recent_hires,
            'marketplace_health': self._assess_marketplace_health()
        }
    
    def recommend_agents_for_research(self, research_description: str, 
                                    max_agents: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend best agents for a specific research problem.
        
        Args:
            research_description: Description of the research problem
            max_agents: Maximum number of agents to recommend
            
        Returns:
            List of agent recommendations with relevance scores
        """
        recommendations = []
        
        for agent in self.available_agents.values():
            # Calculate relevance score
            relevance_score = agent.assess_task_relevance(research_description)
            
            if relevance_score > 0.2:  # Minimum relevance threshold
                # Get performance metrics
                performance = agent.performance_metrics
                
                # Calculate recommendation score (relevance + performance)
                recommendation_score = (
                    relevance_score * 0.6 +
                    performance['average_quality_score'] * 0.25 +
                    performance['success_rate'] * 0.15
                )
                
                recommendations.append({
                    'agent_id': agent.agent_id,
                    'role': agent.role,
                    'expertise': agent.expertise,
                    'relevance_score': relevance_score,
                    'performance_metrics': performance.copy(),
                    'recommendation_score': recommendation_score,
                    'justification': f"High relevance ({relevance_score:.2f}) for {agent.role} expertise"
                })
        
        # Sort by recommendation score and limit results
        recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
        
        return recommendations[:max_agents]
    
    def create_expert_for_domain(self, domain: str, research_context: str = "") -> BaseAgent:
        """
        Dynamically create an expert agent for a specific domain.
        
        Args:
            domain: The domain of expertise needed
            research_context: Additional context about the research problem
            
        Returns:
            Newly created expert agent
        """
        # Generate agent ID
        existing_count = sum(1 for agent in self.agent_registry.values() 
                           if domain.lower() in agent.role.lower())
        agent_id = f"{domain.lower().replace(' ', '_')}_expert_{existing_count + 1}"
        
        # Determine expertise areas based on domain and context
        expertise_areas = self._determine_expertise_areas(domain, research_context)
        
        # Create role description
        role = f"{domain.title()} Expert"
        
        # Create the agent
        new_agent = self._create_expert_agent(agent_id, role, expertise_areas)
        
        # Register the agent
        self.register_agent(new_agent)
        
        logger.info(f"Created new expert agent: {agent_id} for domain: {domain}")
        return new_agent
    
    def _determine_expertise_areas(self, domain: str, context: str = "") -> List[str]:
        """
        Determine specific expertise areas for a domain based on context.
        
        Args:
            domain: Main domain of expertise
            context: Research context to guide expertise selection
            
        Returns:
            List of specific expertise areas
        """
        # Base expertise mapping
        base_expertise = {
            'biology': ['Biology', 'Life Sciences', 'Biological Research'],
            'chemistry': ['Chemistry', 'Chemical Analysis', 'Molecular Science'],
            'physics': ['Physics', 'Physical Sciences', 'Quantitative Analysis'],
            'medicine': ['Medicine', 'Clinical Research', 'Health Sciences'],
            'psychology': ['Psychology', 'Behavioral Science', 'Mental Health'],
            'neuroscience': ['Neuroscience', 'Brain Science', 'Cognitive Science'],
            'computer_science': ['Computer Science', 'Computational Methods', 'Software Engineering'],
            'mathematics': ['Mathematics', 'Mathematical Modeling', 'Statistical Analysis'],
            'engineering': ['Engineering', 'Technical Design', 'Systems Analysis'],
            'social_sciences': ['Social Sciences', 'Human Behavior', 'Society Analysis'],
            'environmental_science': ['Environmental Science', 'Ecology', 'Sustainability'],
            'economics': ['Economics', 'Economic Analysis', 'Market Research']
        }
        
        # Get base expertise or create generic ones
        expertise = base_expertise.get(domain.lower(), [domain.title(), f"{domain.title()} Research"])
        
        # Add context-specific expertise if available
        if context:
            context_lower = context.lower()
            if 'data' in context_lower or 'analysis' in context_lower:
                expertise.append('Data Analysis')
            if 'machine learning' in context_lower or 'ai' in context_lower:
                expertise.append('Machine Learning')
            if 'clinical' in context_lower:
                expertise.append('Clinical Research')
            if 'experimental' in context_lower:
                expertise.append('Experimental Design')
        
        return expertise
        
    def create_specialized_agent(self, role: str, expertise: List[str], 
                               agent_id: Optional[str] = None) -> BaseAgent:
        """
        Create a new specialized agent dynamically.
        
        Args:
            role: Role description for the new agent
            expertise: List of expertise domains
            agent_id: Optional custom agent ID
            
        Returns:
            Newly created agent instance
        """
        if not agent_id:
            agent_id = f"{role.lower().replace(' ', '_')}_{len(self.agent_registry) + 1}"
        
        # Create a generic expert agent with custom role and expertise
        new_agent = self._create_expert_agent(agent_id, role, expertise)
        self.register_agent(new_agent)
        
        logger.info(f"Created specialized agent: {agent_id} with role {role}")
        return new_agent
    
    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()
    
    def _is_recent(self, timestamp: float, hours: int = 24) -> bool:
        """Check if timestamp is within recent hours."""
        import time
        return (time.time() - timestamp) < (hours * 3600)
    
    def _assess_marketplace_health(self) -> str:
        """Assess overall marketplace health."""
        stats = {
            'total': len(self.agent_registry),
            'available': len(self.available_agents),
            'utilization': len(self.hired_agents) / max(1, len(self.agent_registry))
        }
        
        if stats['total'] < 5:
            return "Low - Need more agents"
        elif stats['utilization'] > 0.8:
            return "High demand - Consider adding more agents"
        elif stats['available'] < 2:
            return "Limited availability"
        else:
            return "Healthy"
    
    def list_available_agents(self) -> List[Dict[str, Any]]:
        """List all available agents with their details."""
        return [
            {
                'agent_id': agent.agent_id,
                'role': agent.role,
                'expertise': agent.expertise,
                'performance_metrics': agent.performance_metrics.copy(),
                'status': 'available'
            }
            for agent in self.available_agents.values()
        ]
    
    def list_hired_agents(self) -> List[Dict[str, Any]]:
        """List all currently hired agents with their details."""
        return [
            {
                'agent_id': agent.agent_id,
                'role': agent.role,
                'expertise': agent.expertise,
                'performance_metrics': agent.performance_metrics.copy(),
                'current_task': agent.current_task.get('id') if agent.current_task else None,
                'status': 'hired'
            }
            for agent in self.hired_agents.values()
        ]