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

# Import CLI conditionally to avoid circular imports
try:
    from .cli import CLI
except ImportError:
    CLI = None

# Import visualization function
from .results_visualizer import visualize as ResultsVisualizer
from .specialist_registry import SpecialistRegistry
from .manuscript_drafter import draft as ManuscriptDrafter
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