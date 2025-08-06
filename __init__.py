#!/usr/bin/env python3
"""
AI Research Lab Framework

A comprehensive multi-agent research framework for conducting AI-powered research.
"""

__version__ = "1.0.0"
__author__ = "Expression Neuroscience Institute"

# Core framework components
try:
    from .core import MultiAgentResearchFramework, create_framework, VirtualLabMeetingSystem, AIResearchLab
    from .agents import AgentMarketplace, BaseAgent, DomainExpert, LLMClient, PrincipalInvestigator, ScientificCritic
    from .tools import (
        AccuracyEvaluator, AnalysisTool, BaseTool, CollaborationTool, 
        DynamicToolBuilder, ExperimentalTool, GenericTool, LiteratureTool, ToolRegistry
    )
    from .data import (
        CLI, CostManager, Critic, LiteratureRetriever, ManuscriptDrafter, 
        ResultsVisualizer, SpecialistRegistry
    )
    from .memory import ContextManager, KnowledgeRepository, VectorDatabase
    from .experiments import Experiment
except ImportError:
    # Handle case where this module is imported directly (e.g., by pytest)
    # In this case, we can't use relative imports, so we'll just define the version
    pass

# Export main components for easy access
__all__ = [
    'MultiAgentResearchFramework',
    'create_framework', 
    'VirtualLabMeetingSystem',
    'AIResearchLab',
    'AgentMarketplace',
    'BaseAgent',
    'DomainExpert',
    'LLMClient',
    'PrincipalInvestigator',
    'ScientificCritic',
    'AccuracyEvaluator',
    'AnalysisTool',
    'BaseTool',
    'CollaborationTool',
    'DynamicToolBuilder',
    'ExperimentalTool',
    'GenericTool',
    'LiteratureTool',
    'ToolRegistry',
    'CLI',
    'CostManager',
    'Critic',
    'LiteratureRetriever',
    'ManuscriptDrafter',
    'ResultsVisualizer',
    'SpecialistRegistry',
    'ContextManager',
    'KnowledgeRepository',
    'VectorDatabase',
]