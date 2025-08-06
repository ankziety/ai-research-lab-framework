"""
Core framework components for the AI Research Lab.

This package contains the main framework components including:
- MultiAgentResearchFramework: Main research framework
- VirtualLabMeetingSystem: Virtual lab meeting system
- AIResearchLab: Legacy research lab implementation
"""

# Import main framework components
try:
    from .multi_agent_framework import MultiAgentResearchFramework, create_framework
    from .virtual_lab import VirtualLabMeetingSystem
    from .ai_research_lab import AIResearchLab
except ImportError:
    # Handle case where modules can't be imported due to circular dependencies
    MultiAgentResearchFramework = None
    create_framework = None
    VirtualLabMeetingSystem = None
    AIResearchLab = None

__all__ = [
    'MultiAgentResearchFramework',
    'create_framework',
    'VirtualLabMeetingSystem',
    'AIResearchLab'
] 