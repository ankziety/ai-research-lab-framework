"""
Virtual Lab Agent Marketplace

A marketplace for Virtual Lab agents only. This replaces the existing agent marketplace
to work exclusively with Virtual Lab methodology and agents.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from core.virtual_lab_integration.agent import Agent

logger = logging.getLogger(__name__)


@dataclass
class AgentCategory:
    """Category for organizing Virtual Lab agents."""
    name: str
    description: str
    keywords: List[str]
    agents: List[Agent]


class VirtualLabAgentMarketplace:
    """
    Marketplace for Virtual Lab agents only.
    
    This marketplace manages Virtual Lab agents and provides hiring capabilities
    for research projects using Virtual Lab methodology.
    """
    
    def __init__(self):
        """Initialize the Virtual Lab agent marketplace."""
        self.registered_agents: Dict[str, Agent] = {}
        self.agent_categories: Dict[str, AgentCategory] = {}
        self.agent_performance: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default Virtual Lab agents
        self._initialize_default_agents()
        
        logger.info("Virtual Lab Agent Marketplace initialized")
    
    def _initialize_default_agents(self):
        """Initialize default Virtual Lab agents."""
        
        # Principal Investigator
        pi_agent = Agent(
            title="Principal Investigator",
            expertise="research coordination, project management, scientific leadership",
            goal="Coordinate research teams and ensure scientific rigor",
            role="Research Coordinator",
            model="gpt-4"
        )
        
        # Scientific Critic
        critic_agent = Agent(
            title="Scientific Critic",
            expertise="scientific validation, quality assurance, peer review",
            goal="Ensure scientific rigor and validate research outputs",
            role="Quality Assurance Expert",
            model="gpt-4"
        )
        
        # Research Methodology Expert
        methodology_agent = Agent(
            title="Research Methodology Expert",
            expertise="experimental design, statistical analysis, research methods",
            goal="Design robust research methodologies and experimental protocols",
            role="Methodology Specialist",
            model="gpt-4"
        )
        
        # Literature Research Expert
        literature_agent = Agent(
            title="Literature Research Expert",
            expertise="literature review, citation analysis, academic research",
            goal="Conduct comprehensive literature reviews and identify research gaps",
            role="Literature Specialist",
            model="gpt-4"
        )
        
        # Data Science Expert
        data_agent = Agent(
            title="Data Science Expert",
            expertise="data analysis, machine learning, statistical modeling",
            goal="Analyze data and develop predictive models",
            role="Data Analyst",
            model="gpt-4"
        )
        
        # Critical Analysis Expert
        critical_agent = Agent(
            title="Critical Analysis Expert",
            expertise="logical reasoning, argument analysis, critical thinking",
            goal="Provide critical analysis and identify logical flaws",
            role="Critical Analyst",
            model="gpt-4"
        )
        
        # Register default agents
        self.register_agent(pi_agent)
        self.register_agent(critic_agent)
        self.register_agent(methodology_agent)
        self.register_agent(literature_agent)
        self.register_agent(data_agent)
        self.register_agent(critical_agent)
        
        # Create agent categories
        self._create_agent_categories()
        
        logger.info(f"Initialized {len(self.registered_agents)} default Virtual Lab agents")
    
    def _create_agent_categories(self):
        """Create categories for organizing agents."""
        
        # Leadership category
        leadership_agents = [
            self.registered_agents["Principal Investigator"],
            self.registered_agents["Scientific Critic"]
        ]
        
        self.agent_categories["leadership"] = AgentCategory(
            name="Leadership",
            description="Research coordination and quality assurance",
            keywords=["coordination", "leadership", "quality", "critique"],
            agents=leadership_agents
        )
        
        # Research category
        research_agents = [
            self.registered_agents["Research Methodology Expert"],
            self.registered_agents["Literature Research Expert"]
        ]
        
        self.agent_categories["research"] = AgentCategory(
            name="Research",
            description="Research methodology and literature analysis",
            keywords=["methodology", "literature", "experimental", "review"],
            agents=research_agents
        )
        
        # Analysis category
        analysis_agents = [
            self.registered_agents["Data Science Expert"],
            self.registered_agents["Critical Analysis Expert"]
        ]
        
        self.agent_categories["analysis"] = AgentCategory(
            name="Analysis",
            description="Data analysis and critical thinking",
            keywords=["data", "analysis", "critical", "modeling"],
            agents=analysis_agents
        )
    
    def register_agent(self, agent: Agent) -> bool:
        """
        Register a Virtual Lab agent in the marketplace.
        
        Args:
            agent: Virtual Lab agent to register
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            if agent.title in self.registered_agents:
                logger.warning(f"Agent {agent.title} already registered")
                return False
            
            self.registered_agents[agent.title] = agent
            
            # Initialize performance tracking
            self.agent_performance[agent.title] = {
                'hires': 0,
                'successful_projects': 0,
                'total_projects': 0,
                'average_rating': 0.0,
                'last_hired': None,
                'expertise_areas': agent.expertise.split(', ')
            }
            
            logger.info(f"Registered Virtual Lab agent: {agent.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent.title}: {e}")
            return False
    
    def hire_agents_for_research(self, research_question: str, 
                                constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Hire appropriate Virtual Lab agents for research.
        
        Args:
            research_question: Research question to investigate
            constraints: Research constraints (budget, time, etc.)
            
        Returns:
            Dictionary with hiring results
        """
        constraints = constraints or {}
        
        try:
            # Analyze research question to determine required expertise
            required_expertise = self._analyze_research_requirements(research_question)
            
            # Select agents based on expertise match
            selected_agents = self._select_agents_by_expertise(required_expertise, constraints)
            
            # Update hiring statistics
            for agent_title in selected_agents:
                if agent_title in self.agent_performance:
                    self.agent_performance[agent_title]['hires'] += 1
                    self.agent_performance[agent_title]['last_hired'] = time.time()
            
            hiring_result = {
                'success': True,
                'hired_agents': selected_agents,
                'research_question': research_question,
                'required_expertise': required_expertise,
                'constraints': constraints,
                'hiring_timestamp': time.time()
            }
            
            logger.info(f"Hired {len(selected_agents)} Virtual Lab agents for research")
            return hiring_result
            
        except Exception as e:
            logger.error(f"Failed to hire agents: {e}")
            return {
                'success': False,
                'error': str(e),
                'hired_agents': {},
                'research_question': research_question,
                'constraints': constraints
            }
    
    def _analyze_research_requirements(self, research_question: str) -> List[str]:
        """
        Analyze research question to determine required expertise areas.
        
        Args:
            research_question: Research question to analyze
            
        Returns:
            List of required expertise areas
        """
        required_expertise = []
        
        # Simple keyword-based analysis
        question_lower = research_question.lower()
        
        # Research methodology keywords
        if any(word in question_lower for word in ['method', 'methodology', 'experiment', 'study', 'protocol']):
            required_expertise.append('methodology')
        
        # Literature review keywords
        if any(word in question_lower for word in ['literature', 'review', 'previous', 'existing', 'published']):
            required_expertise.append('literature')
        
        # Data analysis keywords
        if any(word in question_lower for word in ['data', 'analysis', 'statistics', 'model', 'algorithm']):
            required_expertise.append('data_analysis')
        
        # Critical analysis keywords
        if any(word in question_lower for word in ['evaluate', 'assess', 'compare', 'analyze', 'examine']):
            required_expertise.append('critical_analysis')
        
        # Always include leadership for coordination
        required_expertise.append('leadership')
        
        return list(set(required_expertise))
    
    def _select_agents_by_expertise(self, required_expertise: List[str], 
                                  constraints: Dict[str, Any]) -> Dict[str, Agent]:
        """
        Select agents based on required expertise and constraints.
        
        Args:
            required_expertise: List of required expertise areas
            constraints: Research constraints
            
        Returns:
            Dictionary of selected agents
        """
        selected_agents = {}
        max_agents = constraints.get('max_agents', 5)
        
        # Map expertise areas to agent categories
        expertise_to_category = {
            'methodology': 'research',
            'literature': 'research',
            'data_analysis': 'analysis',
            'critical_analysis': 'analysis',
            'leadership': 'leadership'
        }
        
        # Select agents from each required category
        for expertise in required_expertise:
            if len(selected_agents) >= max_agents:
                break
                
            category = expertise_to_category.get(expertise)
            if category and category in self.agent_categories:
                category_agents = self.agent_categories[category].agents
                
                # Select the best agent from this category
                best_agent = self._select_best_agent_from_category(category_agents, constraints)
                if best_agent and best_agent.title not in selected_agents:
                    selected_agents[best_agent.title] = best_agent
        
        return selected_agents
    
    def _select_best_agent_from_category(self, category_agents: List[Agent], 
                                       constraints: Dict[str, Any]) -> Optional[Agent]:
        """
        Select the best agent from a category based on performance and constraints.
        
        Args:
            category_agents: List of agents in the category
            constraints: Research constraints
            
        Returns:
            Best agent from the category
        """
        if not category_agents:
            return None
        
        # Simple selection: choose agent with highest success rate
        best_agent = None
        best_score = -1
        
        for agent in category_agents:
            if agent.title in self.agent_performance:
                performance = self.agent_performance[agent.title]
                
                # Calculate score based on success rate and recent activity
                success_rate = (performance['successful_projects'] / 
                              max(performance['total_projects'], 1))
                
                # Prefer agents with recent activity
                recency_bonus = 0.1 if performance['last_hired'] else 0
                
                score = success_rate + recency_bonus
                
                if score > best_score:
                    best_score = score
                    best_agent = agent
        
        return best_agent or category_agents[0]
    
    def get_agent_performance(self, agent_title: str) -> Optional[Dict[str, Any]]:
        """
        Get performance statistics for an agent.
        
        Args:
            agent_title: Title of the agent
            
        Returns:
            Performance statistics or None if agent not found
        """
        return self.agent_performance.get(agent_title)
    
    def update_agent_performance(self, agent_title: str, project_success: bool, 
                               rating: Optional[float] = None):
        """
        Update agent performance statistics.
        
        Args:
            agent_title: Title of the agent
            project_success: Whether the project was successful
            rating: Optional rating (0.0 to 1.0)
        """
        if agent_title in self.agent_performance:
            performance = self.agent_performance[agent_title]
            performance['total_projects'] += 1
            
            if project_success:
                performance['successful_projects'] += 1
            
            if rating is not None:
                # Update average rating
                current_avg = performance['average_rating']
                total_projects = performance['total_projects']
                performance['average_rating'] = ((current_avg * (total_projects - 1)) + rating) / total_projects
    
    def list_available_agents(self) -> List[Dict[str, Any]]:
        """
        List all available agents with their information.
        
        Returns:
            List of agent information dictionaries
        """
        agents_info = []
        
        for title, agent in self.registered_agents.items():
            performance = self.agent_performance.get(title, {})
            
            agent_info = {
                'title': agent.title,
                'expertise': agent.expertise,
                'goal': agent.goal,
                'role': agent.role,
                'model': agent.model,
                'performance': performance
            }
            
            agents_info.append(agent_info)
        
        return agents_info
    
    def get_agent_by_title(self, title: str) -> Optional[Agent]:
        """
        Get an agent by title.
        
        Args:
            title: Agent title
            
        Returns:
            Agent instance or None if not found
        """
        return self.registered_agents.get(title)
    
    def remove_agent(self, title: str) -> bool:
        """
        Remove an agent from the marketplace.
        
        Args:
            title: Agent title to remove
            
        Returns:
            True if removal successful, False otherwise
        """
        if title in self.registered_agents:
            del self.registered_agents[title]
            
            if title in self.agent_performance:
                del self.agent_performance[title]
            
            logger.info(f"Removed agent: {title}")
            return True
        
        return False
