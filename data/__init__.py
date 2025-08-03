"""
Data management components for the AI Research Lab.

This package contains data processing and management components including:
- LiteratureRetriever: Literature search and retrieval
- CostManager: Cost tracking and management
- CLI: Command line interface
- ResultsVisualizer: Results visualization
- SpecialistRegistry: Specialist agent registry
- ManuscriptDrafter: Manuscript generation
- Critic: Traditional critic implementation
"""

from .literature_retriever import LiteratureRetriever
from .cost_manager import CostManager
from .cli import CLI
from .results_visualizer import ResultsVisualizer
from .specialist_registry import SpecialistRegistry
from .manuscript_drafter import ManuscriptDrafter
from .critic import Critic

__all__ = [
    'LiteratureRetriever',
    'CostManager',
    'CLI',
    'ResultsVisualizer',
    'SpecialistRegistry',
    'ManuscriptDrafter',
    'Critic'
] 