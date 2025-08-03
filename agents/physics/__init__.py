"""
Physics-Specific Agent System

This module provides specialized AI agents for physics research domains,
extending the base agent framework with domain-specific expertise and capabilities.

Inspired by the Virtual Lab methodology, these agents collaborate to conduct
sophisticated physics research across multiple scales and domains.
"""

from .base_physics_agent import BasePhysicsAgent
from .quantum_physics_agent import QuantumPhysicsAgent
from .computational_physics_agent import ComputationalPhysicsAgent
from .experimental_physics_agent import ExperimentalPhysicsAgent
from .materials_physics_agent import MaterialsPhysicsAgent
from .astrophysics_agent import AstrophysicsAgent
from .physics_agent_registry import PhysicsAgentRegistry

__version__ = "1.0.0"
__author__ = "Physics AI Research Team"

__all__ = [
    'BasePhysicsAgent',
    'QuantumPhysicsAgent',
    'ComputationalPhysicsAgent', 
    'ExperimentalPhysicsAgent',
    'MaterialsPhysicsAgent',
    'AstrophysicsAgent',
    'PhysicsAgentRegistry'
]

# Physics domain mappings for agent creation
PHYSICS_DOMAINS = {
    'quantum_physics': {
        'agent_class': 'QuantumPhysicsAgent',
        'keywords': ['quantum', 'qubit', 'superposition', 'entanglement', 'schrödinger'],
        'specializations': ['quantum_mechanics', 'quantum_computing', 'quantum_field_theory']
    },
    'computational_physics': {
        'agent_class': 'ComputationalPhysicsAgent', 
        'keywords': ['simulation', 'numerical', 'modeling', 'computation', 'algorithm'],
        'specializations': ['molecular_dynamics', 'finite_element', 'monte_carlo']
    },
    'experimental_physics': {
        'agent_class': 'ExperimentalPhysicsAgent',
        'keywords': ['experiment', 'measurement', 'detector', 'calibration', 'uncertainty'],
        'specializations': ['experimental_design', 'data_analysis', 'instrumentation']
    },
    'materials_physics': {
        'agent_class': 'MaterialsPhysicsAgent',
        'keywords': ['material', 'crystal', 'solid', 'structure', 'properties'],
        'specializations': ['condensed_matter', 'crystallography', 'materials_science']
    },
    'astrophysics': {
        'agent_class': 'AstrophysicsAgent',
        'keywords': ['cosmic', 'stellar', 'galaxy', 'gravitational', 'cosmology'],
        'specializations': ['stellar_physics', 'cosmology', 'galactic_dynamics']
    }
}

def create_physics_agent(domain: str, agent_id: str, **kwargs):
    """
    Factory function to create physics agents based on domain.
    
    Args:
        domain: Physics domain (quantum_physics, computational_physics, etc.)
        agent_id: Unique identifier for the agent
        **kwargs: Additional configuration parameters
        
    Returns:
        Specialized physics agent instance
    """
    domain_info = PHYSICS_DOMAINS.get(domain)
    if not domain_info:
        raise ValueError(f"Unknown physics domain: {domain}")
    
    agent_class_name = domain_info['agent_class']
    
    # Import and create the appropriate agent class
    if agent_class_name == 'QuantumPhysicsAgent':
        return QuantumPhysicsAgent(agent_id, **kwargs)
    elif agent_class_name == 'ComputationalPhysicsAgent':
        return ComputationalPhysicsAgent(agent_id, **kwargs)
    elif agent_class_name == 'ExperimentalPhysicsAgent':
        return ExperimentalPhysicsAgent(agent_id, **kwargs)
    elif agent_class_name == 'MaterialsPhysicsAgent':
        return MaterialsPhysicsAgent(agent_id, **kwargs)
    elif agent_class_name == 'AstrophysicsAgent':
        return AstrophysicsAgent(agent_id, **kwargs)
    else:
        raise ValueError(f"Unknown agent class: {agent_class_name}")

def get_physics_domain_for_query(query: str) -> str:
    """
    Determine the most relevant physics domain for a research query.
    
    Args:
        query: Research question or query string
        
    Returns:
        Most relevant physics domain identifier
    """
    query_lower = query.lower()
    domain_scores = {}
    
    for domain, domain_info in PHYSICS_DOMAINS.items():
        score = 0
        keywords = domain_info['keywords']
        
        for keyword in keywords:
            if keyword in query_lower:
                score += 1
        
        if score > 0:
            domain_scores[domain] = score
    
    if domain_scores:
        # Return domain with highest keyword match score
        return max(domain_scores, key=domain_scores.get)
    else:
        # Default to computational physics for general physics queries
        return 'computational_physics'