"""Virtual Lab package."""

from .__about__ import __version__
from .agent import Agent

# Conditionally import run_meeting to avoid dependency issues
try:
    from .run_meeting import run_meeting
    _RUN_MEETING_AVAILABLE = True
except ImportError:
    run_meeting = None
    _RUN_MEETING_AVAILABLE = False


__all__ = [
    "__version__",
    "Agent",
]

# Only add run_meeting to __all__ if it's available
if _RUN_MEETING_AVAILABLE:
    __all__.append("run_meeting")
