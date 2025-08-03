"""
Physics Agent Registry - Separate registry for physics-specific agents.

This registry manages physics agents independently from the main agent marketplace
to avoid conflicts while providing integration capabilities.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Type
from enum import Enum

from .base_physics_agent import BasePhysicsAgent, PhysicsScale, PhysicsMethodology

logger = logging.getLogger(__name__)


class PhysicsAgentType(Enum):
    """Types of physics agents available in the registry."""
    QUANTUM_PHYSICS = "quantum_physics"
    COMPUTATIONAL_PHYSICS = "computational_physics" 
    EXPERIMENTAL_PHYSICS = "experimental_physics"
    MATERIALS_PHYSICS = "materials_physics"
    ASTROPHYSICS = "astrophysics"


class PhysicsAgentRegistry:
    """
    Registry for physics-specific agents that works alongside the main agent marketplace.
    
    Provides specialized physics agent management without modifying the existing
    agent framework.
    """
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None, cost_manager=None):
        """
        Initialize physics agent registry.
        
        Args:
            llm_config: Configuration for LLM models
            cost_manager: Optional cost manager for tracking API usage
        """
        self.physics_agents = {}
        self.available_physics_agents = {}
        self.hired_physics_agents = {}
        self.agent_creation_history = []
        self.physics_agent_registry = {}
        self.llm_config = llm_config or {}
        self.cost_manager = cost_manager
        
        # Physics agent type mappings
        self.agent_type_mappings = {
            PhysicsAgentType.QUANTUM_PHYSICS: {
                'class_name': 'QuantumPhysicsAgent',
                'expertise': ['Quantum Mechanics', 'Quantum Computing', 'Quantum Field Theory'],
                'scales': [PhysicsScale.QUANTUM, PhysicsScale.ATOMIC],
                'methodologies': [PhysicsMethodology.THEORETICAL, PhysicsMethodology.COMPUTATIONAL]
            },
            PhysicsAgentType.COMPUTATIONAL_PHYSICS: {
                'class_name': 'ComputationalPhysicsAgent',
                'expertise': ['Numerical Methods', 'Simulation', 'Mathematical Modeling'],
                'scales': [PhysicsScale.NANO, PhysicsScale.MICRO, PhysicsScale.MACRO],
                'methodologies': [PhysicsMethodology.COMPUTATIONAL]
            },
            PhysicsAgentType.EXPERIMENTAL_PHYSICS: {
                'class_name': 'ExperimentalPhysicsAgent',
                'expertise': ['Experimental Design', 'Data Analysis', 'Uncertainty Analysis'],
                'scales': [PhysicsScale.ATOMIC, PhysicsScale.MOLECULAR, PhysicsScale.MACRO],
                'methodologies': [PhysicsMethodology.EXPERIMENTAL]
            },
            PhysicsAgentType.MATERIALS_PHYSICS: {
                'class_name': 'MaterialsPhysicsAgent',
                'expertise': ['Condensed Matter', 'Materials Science', 'Crystal Structures'],
                'scales': [PhysicsScale.ATOMIC, PhysicsScale.NANO, PhysicsScale.MICRO],
                'methodologies': [PhysicsMethodology.THEORETICAL, PhysicsMethodology.EXPERIMENTAL, PhysicsMethodology.COMPUTATIONAL]
            },
            PhysicsAgentType.ASTROPHYSICS: {
                'class_name': 'AstrophysicsAgent',
                'expertise': ['Cosmology', 'Stellar Physics', 'Gravitational Physics'],
                'scales': [PhysicsScale.STELLAR, PhysicsScale.GALACTIC, PhysicsScale.COSMIC],
                'methodologies': [PhysicsMethodology.THEORETICAL, PhysicsMethodology.OBSERVATIONAL]
            }
        }
        
        logger.info("Physics Agent Registry initialized")
    
    def create_physics_agent(self, agent_type: PhysicsAgentType, agent_id: str, 
                            custom_config: Optional[Dict[str, Any]] = None) -> BasePhysicsAgent:
        """
        Create a new physics agent of specified type.
        
        Args:
            agent_type: Type of physics agent to create
            agent_id: Unique identifier for the agent
            custom_config: Optional custom configuration overrides
            
        Returns:
            Created physics agent instance
        """
        if agent_id in self.physics_agent_registry:
            raise ValueError(f"Physics agent {agent_id} already exists")
        
        agent_config = self.agent_type_mappings.get(agent_type)
        if not agent_config:
            raise ValueError(f"Unknown physics agent type: {agent_type}")
        
        # Apply custom configuration if provided
        if custom_config:
            agent_config = {**agent_config, **custom_config}
        
        # Import and create the specific agent class
        agent = self._instantiate_physics_agent(agent_type, agent_id, agent_config)
        
        # Register the agent
        self.register_physics_agent(agent)
        
        # Record creation
        self.agent_creation_history.append({
            'agent_id': agent_id,
            'agent_type': agent_type.value,
            'created_at': time.time(),
            'config': agent_config
        })
        
        logger.info(f"Created physics agent: {agent_id} ({agent_type.value})")
        return agent
    
    def register_physics_agent(self, agent: BasePhysicsAgent) -> None:
        """
        Register a physics agent in the registry.
        
        Args:
            agent: Physics agent instance to register
        """
        if agent.agent_id in self.physics_agent_registry:
            raise ValueError(f"Physics agent {agent.agent_id} already registered")
        
        self.physics_agent_registry[agent.agent_id] = agent
        self.available_physics_agents[agent.agent_id] = agent
        
        logger.info(f"Registered physics agent: {agent.agent_id} ({agent.physics_domain})")
    
    def hire_physics_agent(self, agent_id: str) -> bool:
        """
        Hire a physics agent (move from available to hired).
        
        Args:
            agent_id: ID of physics agent to hire
            
        Returns:
            True if successfully hired, False otherwise
        """
        if agent_id not in self.available_physics_agents:
            logger.warning(f"Physics agent {agent_id} not available for hiring")
            return False
        
        agent = self.available_physics_agents.pop(agent_id)
        self.hired_physics_agents[agent_id] = agent
        
        logger.info(f"Hired physics agent: {agent_id}")
        return True
    
    def release_physics_agent(self, agent_id: str) -> bool:
        """
        Release a physics agent (move from hired back to available).
        
        Args:
            agent_id: ID of physics agent to release
            
        Returns:
            True if successfully released, False otherwise
        """
        if agent_id not in self.hired_physics_agents:
            logger.warning(f"Physics agent {agent_id} not currently hired")
            return False
        
        agent = self.hired_physics_agents.pop(agent_id)
        self.available_physics_agents[agent_id] = agent
        
        logger.info(f"Released physics agent: {agent_id}")
        return True
    
    def get_physics_agent(self, agent_id: str) -> Optional[BasePhysicsAgent]:
        """
        Get a physics agent by ID.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Physics agent instance or None if not found
        """
        return self.physics_agent_registry.get(agent_id)
    
    def find_physics_agents_by_domain(self, physics_domain: str) -> List[BasePhysicsAgent]:
        """
        Find physics agents by their physics domain.
        
        Args:
            physics_domain: Physics domain to search for
            
        Returns:
            List of agents with matching domain
        """
        matching_agents = []
        
        for agent in self.physics_agent_registry.values():
            if agent.physics_domain == physics_domain:
                matching_agents.append(agent)
        
        return matching_agents
    
    def find_physics_agents_by_scale(self, scale: PhysicsScale) -> List[BasePhysicsAgent]:
        """
        Find physics agents that work at a specific physical scale.
        
        Args:
            scale: Physical scale to search for
            
        Returns:
            List of agents that work at the specified scale
        """
        matching_agents = []
        
        for agent in self.physics_agent_registry.values():
            if scale in agent.physics_scales:
                matching_agents.append(agent)
        
        return matching_agents
    
    def find_physics_agents_by_methodology(self, methodology: PhysicsMethodology) -> List[BasePhysicsAgent]:
        """
        Find physics agents that use a specific methodology.
        
        Args:
            methodology: Physics methodology to search for
            
        Returns:
            List of agents that use the specified methodology
        """
        matching_agents = []
        
        for agent in self.physics_agent_registry.values():
            if methodology in agent.physics_methodologies:
                matching_agents.append(agent)
        
        return matching_agents
    
    def recommend_physics_agents_for_research(self, research_question: str, 
                                            max_agents: int = 3) -> List[Dict[str, Any]]:
        """
        Recommend best physics agents for a research question.
        
        Args:
            research_question: Physics research question
            max_agents: Maximum number of agents to recommend
            
        Returns:
            List of agent recommendations with relevance scores
        """
        recommendations = []
        
        for agent in self.available_physics_agents.values():
            # Calculate relevance score
            relevance_score = self._calculate_physics_relevance(agent, research_question)
            
            if relevance_score > 0.3:  # Minimum relevance threshold
                recommendations.append({
                    'agent_id': agent.agent_id,
                    'agent_type': agent.physics_domain,
                    'relevance_score': relevance_score,
                    'expertise': agent.expertise,
                    'scales': [scale.value for scale in agent.physics_scales],
                    'methodologies': [method.value for method in agent.physics_methodologies],
                    'complexity_score': self._assess_agent_complexity_capability(agent),
                    'justification': self._generate_recommendation_justification(agent, research_question)
                })
        
        # Sort by relevance score
        recommendations.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return recommendations[:max_agents]
    
    def get_physics_agent_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about physics agents in the registry.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_physics_agents': len(self.physics_agent_registry),
            'available_physics_agents': len(self.available_physics_agents),
            'hired_physics_agents': len(self.hired_physics_agents),
            'agents_by_domain': {},
            'agents_by_scale': {},
            'agents_by_methodology': {},
            'creation_history_count': len(self.agent_creation_history),
            'utilization_rate': 0.0
        }
        
        # Calculate utilization rate
        if stats['total_physics_agents'] > 0:
            stats['utilization_rate'] = stats['hired_physics_agents'] / stats['total_physics_agents']
        
        # Count agents by domain
        for agent in self.physics_agent_registry.values():
            domain = agent.physics_domain
            stats['agents_by_domain'][domain] = stats['agents_by_domain'].get(domain, 0) + 1
        
        # Count agents by scale
        for agent in self.physics_agent_registry.values():
            for scale in agent.physics_scales:
                scale_name = scale.value
                stats['agents_by_scale'][scale_name] = stats['agents_by_scale'].get(scale_name, 0) + 1
        
        # Count agents by methodology
        for agent in self.physics_agent_registry.values():
            for methodology in agent.physics_methodologies:
                method_name = methodology.value
                stats['agents_by_methodology'][method_name] = stats['agents_by_methodology'].get(method_name, 0) + 1
        
        return stats
    
    def create_physics_agent_team(self, research_question: str, 
                                 team_size: int = 3) -> List[BasePhysicsAgent]:
        """
        Create a specialized physics agent team for a research question.
        
        Args:
            research_question: Physics research question
            team_size: Desired team size
            
        Returns:
            List of hired physics agents forming the team
        """
        # Get recommendations
        recommendations = self.recommend_physics_agents_for_research(
            research_question, max_agents=team_size * 2
        )
        
        # Create and hire agents as needed
        team = []
        agent_counter = 0
        
        for rec in recommendations[:team_size]:
            agent_id = rec['agent_id']
            
            # If agent exists and is available, hire it
            if agent_id in self.available_physics_agents:
                if self.hire_physics_agent(agent_id):
                    team.append(self.hired_physics_agents[agent_id])
            else:
                # Create new agent of the recommended type
                agent_type_str = rec['agent_type']
                agent_type = self._get_agent_type_from_domain(agent_type_str)
                
                if agent_type:
                    new_agent_id = f"physics_team_{agent_type_str}_{agent_counter}"
                    new_agent = self.create_physics_agent(agent_type, new_agent_id)
                    
                    if self.hire_physics_agent(new_agent_id):
                        team.append(self.hired_physics_agents[new_agent_id])
                    
                    agent_counter += 1
        
        logger.info(f"Created physics agent team of {len(team)} agents for research")
        return team
    
    def list_available_physics_agents(self) -> List[Dict[str, Any]]:
        """List all available physics agents with their details."""
        return [
            {
                'agent_id': agent.agent_id,
                'agent_type': agent.physics_domain,
                'role': agent.role,
                'expertise': agent.expertise,
                'scales': [scale.value for scale in agent.physics_scales],
                'methodologies': [method.value for method in agent.physics_methodologies],
                'status': 'available'
            }
            for agent in self.available_physics_agents.values()
        ]
    
    def list_hired_physics_agents(self) -> List[Dict[str, Any]]:
        """List all currently hired physics agents with their details."""
        return [
            {
                'agent_id': agent.agent_id,
                'agent_type': agent.physics_domain,
                'role': agent.role,
                'expertise': agent.expertise,
                'scales': [scale.value for scale in agent.physics_scales],
                'methodologies': [method.value for method in agent.physics_methodologies],
                'status': 'hired'
            }
            for agent in self.hired_physics_agents.values()
        ]
    
    # Private helper methods
    
    def _instantiate_physics_agent(self, agent_type: PhysicsAgentType, 
                                  agent_id: str, agent_config: Dict[str, Any]) -> BasePhysicsAgent:
        """Instantiate a specific physics agent type."""
        from . import (
            QuantumPhysicsAgent, ComputationalPhysicsAgent, ExperimentalPhysicsAgent,
            MaterialsPhysicsAgent, AstrophysicsAgent
        )
        
        role = f"{agent_type.value.replace('_', ' ').title()} Expert"
        expertise = agent_config.get('expertise', [])
        
        if agent_type == PhysicsAgentType.QUANTUM_PHYSICS:
            return QuantumPhysicsAgent(agent_id, role, expertise, self.llm_config, self.cost_manager)
        elif agent_type == PhysicsAgentType.COMPUTATIONAL_PHYSICS:
            return ComputationalPhysicsAgent(agent_id, role, expertise, self.llm_config, self.cost_manager)
        elif agent_type == PhysicsAgentType.EXPERIMENTAL_PHYSICS:
            return ExperimentalPhysicsAgent(agent_id, role, expertise, self.llm_config, self.cost_manager)
        elif agent_type == PhysicsAgentType.MATERIALS_PHYSICS:
            return MaterialsPhysicsAgent(agent_id, role, expertise, self.llm_config, self.cost_manager)
        elif agent_type == PhysicsAgentType.ASTROPHYSICS:
            return AstrophysicsAgent(agent_id, role, expertise, self.llm_config, self.cost_manager)
        else:
            raise ValueError(f"Cannot instantiate agent type: {agent_type}")
    
    def _calculate_physics_relevance(self, agent: BasePhysicsAgent, 
                                   research_question: str) -> float:
        """Calculate how relevant an agent is to a research question."""
        # Use the agent's own relevance assessment
        base_relevance = agent.assess_task_relevance(research_question)
        
        # Add physics-specific scoring
        physics_boost = 0.0
        
        # Check for domain-specific keywords
        domain_keywords = {
            'quantum_physics': ['quantum', 'qubit', 'superposition', 'entanglement'],
            'computational_physics': ['simulation', 'numerical', 'computational', 'modeling'],
            'experimental_physics': ['experiment', 'measurement', 'data', 'analysis'],
            'materials_physics': ['material', 'crystal', 'solid', 'structure'],
            'astrophysics': ['cosmic', 'stellar', 'galaxy', 'universe']
        }
        
        question_lower = research_question.lower()
        
        if agent.physics_domain in domain_keywords:
            keywords = domain_keywords[agent.physics_domain]
            matches = sum(1 for keyword in keywords if keyword in question_lower)
            physics_boost = matches * 0.1
        
        return min(1.0, base_relevance + physics_boost)
    
    def _assess_agent_complexity_capability(self, agent: BasePhysicsAgent) -> float:
        """Assess an agent's capability to handle complex physics problems."""
        complexity_score = 0.0
        
        # Base score from number of methodologies
        complexity_score += len(agent.physics_methodologies) * 0.2
        
        # Base score from number of scales
        complexity_score += len(agent.physics_scales) * 0.15
        
        # Base score from expertise breadth
        complexity_score += len(agent.expertise) * 0.1
        
        # Domain-specific complexity bonuses
        domain_complexity = {
            'quantum_physics': 0.9,
            'astrophysics': 0.8,
            'computational_physics': 0.7,
            'materials_physics': 0.6,
            'experimental_physics': 0.5
        }
        
        complexity_score += domain_complexity.get(agent.physics_domain, 0.5)
        
        return min(1.0, complexity_score)
    
    def _generate_recommendation_justification(self, agent: BasePhysicsAgent, 
                                             research_question: str) -> str:
        """Generate justification for recommending an agent."""
        justification = f"Agent specializes in {agent.physics_domain} with expertise in {', '.join(agent.expertise)}. "
        
        # Add scale justification
        scales = [scale.value for scale in agent.physics_scales]
        justification += f"Covers physical scales: {', '.join(scales)}. "
        
        # Add methodology justification
        methodologies = [method.value for method in agent.physics_methodologies]
        justification += f"Uses methodologies: {', '.join(methodologies)}."
        
        return justification
    
    def _get_agent_type_from_domain(self, domain: str) -> Optional[PhysicsAgentType]:
        """Get agent type enum from domain string."""
        domain_mapping = {
            'quantum_physics': PhysicsAgentType.QUANTUM_PHYSICS,
            'computational_physics': PhysicsAgentType.COMPUTATIONAL_PHYSICS,
            'experimental_physics': PhysicsAgentType.EXPERIMENTAL_PHYSICS,
            'materials_physics': PhysicsAgentType.MATERIALS_PHYSICS,
            'astrophysics': PhysicsAgentType.ASTROPHYSICS
        }
        
        return domain_mapping.get(domain)