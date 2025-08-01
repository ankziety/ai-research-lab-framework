"""
AI-Powered Research Framework

A comprehensive framework for AI-powered research workflows that integrates experiment 
execution, literature retrieval, manuscript drafting, result visualization, 
and research critique capabilities across any research domain.
"""

from .ai_research_lab import AIPoweredResearchFramework, create_framework

# Import individual components for direct access if needed
from .manuscript_drafter import draft as draft_manuscript
from .literature_retriever import LiteratureRetriever
from .critic import Critic  
from .results_visualizer import visualize as visualize_results
from .specialist_registry import SpecialistRegistry
from .experiments.experiment import ExperimentRunner

__version__ = "1.0.0"
__author__ = "AI-Powered Research Framework Team"

# Main framework exports
__all__ = [
    'AIPoweredResearchFramework',
    'create_framework',
    'LiteratureRetriever',
    'Critic',
    'SpecialistRegistry', 
    'ExperimentRunner',
    'draft_manuscript',
    'visualize_results'
]

# Convenience function for quick framework creation
def framework(config=None):
    """
    Create an AI-Powered Research Framework instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured AIPoweredResearchFramework instance
    """
    return create_framework(config)


# Default configuration template
DEFAULT_CONFIG = {
    'experiment_db_path': 'experiments/experiments.db',
    'output_dir': 'output',
    'manuscript_dir': 'manuscripts', 
    'visualization_dir': 'visualizations',
    'literature_api_url': None,
    'literature_api_key': None,
    'max_literature_results': 10,
    'auto_visualize': True,
    'auto_critique': True
}