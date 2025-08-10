"""
Top-level aggregator package for AI Research Lab Framework.

Provides a stable import surface: `from ai_research_lab import create_framework, MultiAgentResearchFramework, ...`.

Note: This module re-exports objects from top-level packages (`core`, `agents`,
`data`, `tools`, `memory`, `experiments`). Keep imports absolute to avoid
creating artificial package nesting.
"""

__version__ = "1.0.0"

# Core framework components
try:
    from core import (
        MultiAgentResearchFramework,
        create_framework,
        VirtualLabMeetingSystem,
        AIResearchLab,
    )

    from agents import (
        AgentMarketplace,
        BaseAgent,
        PrincipalInvestigatorAgent,
        ScientificCriticAgent,
    )
    from agents.llm_client import LLMClient, get_llm_client

    from tools import (
        ToolRegistry,
        BaseTool,
        ExperimentRunner,
        DataCollector,
        StatisticalAnalyzer,
        DataVisualizer,
        PatternDetector,
        HypothesisValidator,
        LiteratureSearchTool,
        CitationAnalyzer,
        TeamCommunication,
        TaskCoordinator,
    )

    from data import (
        CLI,
        CostManager,
        Critic,
        LiteratureRetriever,
        ManuscriptDrafter,
        ResultsVisualizer,
        SpecialistRegistry,
    )

    from memory import ContextManager, KnowledgeRepository, VectorDatabase

    # Backward-compatible alias for experiments
    from experiments import ExperimentRunner as Experiment
except Exception:
    # Allow import in constrained environments; symbols may be None if imports fail
    pass

__all__ = [
    'MultiAgentResearchFramework',
    'create_framework',
    'VirtualLabMeetingSystem',
    'AIResearchLab',
    'AgentMarketplace',
    'BaseAgent',
    'PrincipalInvestigatorAgent',
    'ScientificCriticAgent',
    'LLMClient',
    'get_llm_client',
    'ToolRegistry',
    'BaseTool',
    'ExperimentRunner',
    'DataCollector',
    'StatisticalAnalyzer',
    'DataVisualizer',
    'PatternDetector',
    'HypothesisValidator',
    'LiteratureSearchTool',
    'CitationAnalyzer',
    'TeamCommunication',
    'TaskCoordinator',
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
    'Experiment',
]


