"""
Core framework components for the AI Research Lab.

This package contains the main framework components including:
- MultiAgentResearchFramework: Main research framework
- VirtualLabMeetingSystem: Virtual lab meeting system
- AIResearchLab: Legacy research lab implementation
"""

from .multi_agent_framework import MultiAgentResearchFramework, create_framework
from .virtual_lab import VirtualLabMeetingSystem
from .ai_research_lab import AIResearchLab

__all__ = [
    'MultiAgentResearchFramework',
    'create_framework',
    'VirtualLabMeetingSystem',
    'AIResearchLab'
] 