"""
Research Tools Module

Provides a comprehensive toolkit for AI agents to conduct research experiments,
data analysis, and other scientific tasks.
"""

from .tool_registry import ToolRegistry
from .base_tool import BaseTool
from .experimental_tools import ExperimentRunner, DataCollector, StatisticalAnalyzer
from .analysis_tools import DataVisualizer, PatternDetector, HypothesisValidator
from .literature_tools import LiteratureSearchTool, CitationAnalyzer
from .collaboration_tools import TeamCommunication, TaskCoordinator

__all__ = [
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
    'TaskCoordinator'
]