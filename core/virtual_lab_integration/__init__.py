"""Virtual Lab package."""

from .__about__ import __version__
from .agent import Agent
from .run_meeting import run_meeting


__all__ = [
    "__version__",
    "Agent",
    "run_meeting",
]
