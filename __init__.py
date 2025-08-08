"""
AI Research Lab Framework

A comprehensive multi-agent research framework for conducting AI-powered research.
"""

__version__ = "1.0.0"
__author__ = "AI Research Lab Team"

# Core framework components
try:
    from core import MultiAgentResearchFramework, create_framework, VirtualLabMeetingSystem, AIResearchLab
except ImportError:
    from .core import MultiAgentResearchFramework, create_framework, VirtualLabMeetingSystem, AIResearchLab

# Data management components
try:
    from data import (
        LiteratureRetriever,
        CostManager,
        ResultsVisualizer,
        SpecialistRegistry,
        ManuscriptDrafter,
        Critic
    )
except ImportError:
    from .data import (
        LiteratureRetriever,
        CostManager,
        ResultsVisualizer,
        SpecialistRegistry,
        ManuscriptDrafter,
        Critic
    )

# Agent system
try:
    from agents import BaseAgent, PrincipalInvestigatorAgent, ScientificCriticAgent, AgentMarketplace
except ImportError:
    from .agents import BaseAgent, PrincipalInvestigatorAgent, ScientificCriticAgent, AgentMarketplace

# Physics-specific agents (NEW)
try:
    from .agents.physics import (
        BasePhysicsAgent, QuantumPhysicsAgent, ComputationalPhysicsAgent,
        ExperimentalPhysicsAgent, MaterialsPhysicsAgent, AstrophysicsAgent,
        PhysicsAgentRegistry, create_physics_agent, get_physics_domain_for_query
    )
    PHYSICS_AGENTS_AVAILABLE = True
except ImportError:
    PHYSICS_AGENTS_AVAILABLE = False

# Memory system
try:
    from memory import VectorDatabase, ContextManager, KnowledgeRepository
except ImportError:
    from .memory import VectorDatabase, ContextManager, KnowledgeRepository

# Tools system
try:
    from tools import ToolRegistry, BaseTool
except ImportError:
    from .tools import ToolRegistry, BaseTool

# Experiments
try:
    from experiments import ExperimentRunner
except ImportError:
    from .experiments import ExperimentRunner

# Convenience functions
def draft_manuscript(results, context):
    """Draft a manuscript from research results."""
    drafter = ManuscriptDrafter()
    return drafter.draft(results, context)

def visualize_results(results, out_path):
    """Visualize research results."""
    visualizer = ResultsVisualizer()
    return visualizer.visualize(results, out_path)

# Main framework creation
def create_research_framework(config=None):
    """Create a new research framework instance."""
    return create_framework(config)

# Physics agent creation convenience function
def create_physics_research_team(research_question, team_size=3):
    """Create a physics research team for a specific question."""
    if PHYSICS_AGENTS_AVAILABLE:
        registry = PhysicsAgentRegistry()
        return registry.create_physics_agent_team(research_question, team_size)
    else:
        raise ImportError("Physics agents not available")

__all__ = [
    'MultiAgentResearchFramework',
    'create_framework',
    'create_research_framework',
    'VirtualLabMeetingSystem',
    'AIResearchLab',
    'LiteratureRetriever',
    'CostManager',
    'ResultsVisualizer',
    'SpecialistRegistry',
    'ManuscriptDrafter',
    'Critic',
    'BaseAgent',
    'PrincipalInvestigatorAgent',
    'ScientificCriticAgent',
    'AgentMarketplace',
    'VectorDatabase',
    'ContextManager',
    'KnowledgeRepository',
    'ToolRegistry',
    'BaseTool',
    'ExperimentRunner',
    'draft_manuscript',
    'visualize_results',
    'create_physics_research_team',
    'PHYSICS_AGENTS_AVAILABLE'
]

# Add physics agents to __all__ if available
if PHYSICS_AGENTS_AVAILABLE:
    __all__.extend([
        'BasePhysicsAgent',
        'QuantumPhysicsAgent',
        'ComputationalPhysicsAgent', 
        'ExperimentalPhysicsAgent',
        'MaterialsPhysicsAgent',
        'AstrophysicsAgent',
        'PhysicsAgentRegistry',
        'create_physics_agent',
        'get_physics_domain_for_query'
    ])