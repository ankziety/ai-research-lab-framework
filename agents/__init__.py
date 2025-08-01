"""
Multi-Agent System for AI-Powered Research Framework
"""

from .base_agent import BaseAgent
from .principal_investigator import PrincipalInvestigatorAgent
from .scientific_critic import ScientificCriticAgent
from .agent_marketplace import AgentMarketplace

__all__ = [
    'BaseAgent',
    'PrincipalInvestigatorAgent', 
    'ScientificCriticAgent',
    'AgentMarketplace'
]