"""
AI Research Lab Framework

A comprehensive multi-agent research framework for conducting AI-powered research.
"""

__version__ = "1.0.0"
__author__ = "AI Research Lab Team"

# Core framework components
from .core import MultiAgentResearchFramework, create_framework, VirtualLabMeetingSystem, AIResearchLab

# Data management components
from .data import (
    LiteratureRetriever,
    CostManager,
    CLI,
    ResultsVisualizer,
    SpecialistRegistry,
    ManuscriptDrafter,
    Critic
)

# Agent system
from .agents import BaseAgent, PrincipalInvestigatorAgent, ScientificCriticAgent, AgentMarketplace

# Memory system
from .memory import VectorDatabase, ContextManager, KnowledgeRepository

# Tools system
from .tools import ToolRegistry, BaseTool

# Experiments
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

__all__ = [
    'MultiAgentResearchFramework',
    'create_framework',
    'create_research_framework',
    'VirtualLabMeetingSystem',
    'AIResearchLab',
    'LiteratureRetriever',
    'CostManager',
    'CLI',
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
    'visualize_results'
]