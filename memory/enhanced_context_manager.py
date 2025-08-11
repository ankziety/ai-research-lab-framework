"""
Enhanced Context Manager with Persistent Cross-Agent Context

Implements persistent context management with:
- process_thought integration for reasoning traces
- vibe_check integration for self-assessment
- State continuity between research sessions
- Cross-agent context sharing and persistence
- Enhanced memory management with vector storage
"""

import logging
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path

from .context_manager import ContextManager
from .vector_database import VectorDatabase

logger = logging.getLogger(__name__)


@dataclass
class ThoughtEntry:
    """Entry for process_thought reasoning traces."""
    thought_number: int
    total_thoughts: int
    thought: str
    stage: str
    timestamp: float
    agent_id: str
    session_id: str
    metadata: Dict[str, Any]


@dataclass
class VibeCheckEntry:
    """Entry for vibe_check self-assessments."""
    goal: str
    plan: str
    progress: str
    uncertainties: List[str]
    task_context: str
    timestamp: float
    agent_id: str
    session_id: str
    assessment_score: float


@dataclass
class AgentContext:
    """Context for a specific agent in a session."""
    agent_id: str
    session_id: str
    role: str
    expertise: List[str]
    current_task: Optional[str]
    task_history: List[Dict[str, Any]]
    context_data: Dict[str, Any]
    last_updated: float
    importance_score: float


class EnhancedContextManager:
    """
    Enhanced context manager with persistent cross-agent context.
    
    Features:
    - process_thought integration for reasoning traces
    - vibe_check integration for self-assessment
    - State continuity between research sessions
    - Cross-agent context sharing and persistence
    - Enhanced memory management with vector storage
    """
    
    def __init__(self, vector_db: VectorDatabase, 
                 base_context_manager: ContextManager,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced context manager.
        
        Args:
            vector_db: Vector database for storage
            base_context_manager: Base context manager
            config: Configuration dictionary
        """
        self.vector_db = vector_db
        self.base_context_manager = base_context_manager
        self.config = config or {}
        
        # Enhanced storage
        self.thought_history = {}  # session_id -> List[ThoughtEntry]
        self.vibe_check_history = {}  # session_id -> List[VibeCheckEntry]
        self.agent_contexts = {}  # agent_id -> AgentContext
        self.session_states = {}  # session_id -> Dict[str, Any]
        
        # Persistence settings
        self.persistence_dir = Path(self.config.get('persistence_dir', 'memory/persistent_context'))
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing persistent data
        self._load_persistent_data()
        
        logger.info("Enhanced Context Manager initialized")
    
    def add_thought(self, session_id: str, agent_id: str, thought: str, 
                   thought_number: int, total_thoughts: int, stage: str,
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a thought entry to the reasoning trace.
        
        Args:
            session_id: Session identifier
            agent_id: Agent identifier
            thought: Thought content
            thought_number: Sequential thought number
            total_thoughts: Total expected thoughts
            stage: Thinking stage
            metadata: Additional metadata
            
        Returns:
            True if thought was added successfully
        """
        if session_id not in self.thought_history:
            self.thought_history[session_id] = []
        
        thought_entry = ThoughtEntry(
            thought_number=thought_number,
            total_thoughts=total_thoughts,
            thought=thought,
            stage=stage,
            timestamp=time.time(),
            agent_id=agent_id,
            session_id=session_id,
            metadata=metadata or {}
        )
        
        self.thought_history[session_id].append(thought_entry)
        
        # Store in vector database for retrieval
        self.vector_db.store_content(
            content=thought,
            content_type="process_thought",
            agent_id=agent_id,
            session_id=session_id,
            importance_score=0.8,
            metadata={
                'thought_number': thought_number,
                'total_thoughts': total_thoughts,
                'stage': stage,
                **(metadata or {})
            }
        )
        
        # Update agent context
        self._update_agent_context(agent_id, session_id, {
            'last_thought': thought,
            'thought_count': len(self.thought_history[session_id]),
            'current_stage': stage
        })
        
        logger.debug(f"Added thought {thought_number}/{total_thoughts} for agent {agent_id} in session {session_id}")
        return True
    
    def add_vibe_check(self, session_id: str, agent_id: str, goal: str, plan: str,
                      progress: str, uncertainties: List[str], task_context: str,
                      assessment_score: float = 0.0) -> bool:
        """
        Add a vibe_check self-assessment entry.
        
        Args:
            session_id: Session identifier
            agent_id: Agent identifier
            goal: Current goal
            plan: Current plan
            progress: Progress description
            uncertainties: List of uncertainties
            task_context: Task context
            assessment_score: Self-assessment score
            
        Returns:
            True if vibe_check was added successfully
        """
        if session_id not in self.vibe_check_history:
            self.vibe_check_history[session_id] = []
        
        vibe_check_entry = VibeCheckEntry(
            goal=goal,
            plan=plan,
            progress=progress,
            uncertainties=uncertainties,
            task_context=task_context,
            timestamp=time.time(),
            agent_id=agent_id,
            session_id=session_id,
            assessment_score=assessment_score
        )
        
        self.vibe_check_history[session_id].append(vibe_check_entry)
        
        # Store in vector database
        self.vector_db.store_content(
            content=f"Goal: {goal}\nPlan: {plan}\nProgress: {progress}\nUncertainties: {uncertainties}",
            content_type="vibe_check",
            agent_id=agent_id,
            session_id=session_id,
            importance_score=0.9,
            metadata={
                'assessment_score': assessment_score,
                'uncertainties_count': len(uncertainties)
            }
        )
        
        # Update agent context
        self._update_agent_context(agent_id, session_id, {
            'last_vibe_check': assessment_score,
            'current_goal': goal,
            'uncertainties': uncertainties
        })
        
        logger.debug(f"Added vibe_check for agent {agent_id} in session {session_id}")
        return True
    
    def get_agent_context(self, agent_id: str, session_id: str) -> Optional[AgentContext]:
        """
        Get context for a specific agent in a session.
        
        Args:
            agent_id: Agent identifier
            session_id: Session identifier
            
        Returns:
            Agent context or None if not found
        """
        context_key = f"{agent_id}_{session_id}"
        return self.agent_contexts.get(context_key)
    
    def update_agent_context(self, agent_id: str, session_id: str, 
                           context_data: Dict[str, Any]) -> bool:
        """
        Update context for a specific agent.
        
        Args:
            agent_id: Agent identifier
            session_id: Session identifier
            context_data: Context data to update
            
        Returns:
            True if context was updated successfully
        """
        return self._update_agent_context(agent_id, session_id, context_data)
    
    def _update_agent_context(self, agent_id: str, session_id: str, 
                            context_data: Dict[str, Any]) -> bool:
        """Internal method to update agent context."""
        context_key = f"{agent_id}_{session_id}"
        
        if context_key not in self.agent_contexts:
            # Create new agent context
            self.agent_contexts[context_key] = AgentContext(
                agent_id=agent_id,
                session_id=session_id,
                role="Unknown",
                expertise=[],
                current_task=None,
                task_history=[],
                context_data={},
                last_updated=time.time(),
                importance_score=0.5
            )
        
        # Update context data
        agent_context = self.agent_contexts[context_key]
        agent_context.context_data.update(context_data)
        agent_context.last_updated = time.time()
        
        return True
    
    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """
        Get complete state for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session state dictionary
        """
        if session_id not in self.session_states:
            self.session_states[session_id] = {
                'session_id': session_id,
                'created_at': time.time(),
                'last_updated': time.time(),
                'agents': {},
                'thoughts': [],
                'vibe_checks': [],
                'context_summary': {}
            }
        
        # Update with current data
        session_state = self.session_states[session_id]
        session_state['last_updated'] = time.time()
        session_state['thoughts'] = self.thought_history.get(session_id, [])
        session_state['vibe_checks'] = self.vibe_check_history.get(session_id, [])
        
        # Add agent contexts
        for context_key, agent_context in self.agent_contexts.items():
            if agent_context.session_id == session_id:
                session_state['agents'][agent_context.agent_id] = asdict(agent_context)
        
        return session_state
    
    def get_cross_agent_context(self, session_id: str, 
                               query: str = None) -> Dict[str, Any]:
        """
        Get cross-agent context for a session.
        
        Args:
            session_id: Session identifier
            query: Optional query for relevant context retrieval
            
        Returns:
            Cross-agent context dictionary
        """
        # Get all agent contexts for the session
        agent_contexts = {}
        for context_key, agent_context in self.agent_contexts.items():
            if agent_context.session_id == session_id:
                agent_contexts[agent_context.agent_id] = asdict(agent_context)
        
        # Get relevant thoughts and vibe_checks
        thoughts = self.thought_history.get(session_id, [])
        vibe_checks = self.vibe_check_history.get(session_id, [])
        
        # If query provided, get relevant context from vector database
        relevant_context = []
        if query:
            relevant_context = self.vector_db.search_similar(
                query=query,
                limit=5,
                session_id=session_id,
                min_importance=0.3
            )
        
        return {
            'session_id': session_id,
            'agent_contexts': agent_contexts,
            'thoughts_summary': self._summarize_thoughts(thoughts),
            'vibe_checks_summary': self._summarize_vibe_checks(vibe_checks),
            'relevant_context': relevant_context,
            'session_duration': self._calculate_session_duration(session_id),
            'agent_activity': self._get_agent_activity(session_id)
        }
    
    def persist_session_state(self, session_id: str) -> bool:
        """
        Persist session state to disk.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if state was persisted successfully
        """
        try:
            session_state = self.get_session_state(session_id)
            
            # Create session directory
            session_dir = self.persistence_dir / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            
            # Save session state
            state_file = session_dir / "session_state.json"
            with open(state_file, 'w') as f:
                json.dump(session_state, f, indent=2, default=str)
            
            # Save thoughts
            thoughts_file = session_dir / "thoughts.json"
            thoughts = self.thought_history.get(session_id, [])
            with open(thoughts_file, 'w') as f:
                json.dump([asdict(thought) for thought in thoughts], f, indent=2, default=str)
            
            # Save vibe_checks
            vibe_checks_file = session_dir / "vibe_checks.json"
            vibe_checks = self.vibe_check_history.get(session_id, [])
            with open(vibe_checks_file, 'w') as f:
                json.dump([asdict(vibe_check) for vibe_check in vibe_checks], f, indent=2, default=str)
            
            logger.info(f"Persisted session state for {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to persist session state for {session_id}: {e}")
            return False
    
    def restore_session_state(self, session_id: str) -> bool:
        """
        Restore session state from disk.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if state was restored successfully
        """
        try:
            session_dir = self.persistence_dir / session_id
            
            if not session_dir.exists():
                logger.warning(f"Session directory not found: {session_dir}")
                return False
            
            # Restore session state
            state_file = session_dir / "session_state.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    session_state = json.load(f)
                self.session_states[session_id] = session_state
            
            # Restore thoughts
            thoughts_file = session_dir / "thoughts.json"
            if thoughts_file.exists():
                with open(thoughts_file, 'r') as f:
                    thoughts_data = json.load(f)
                self.thought_history[session_id] = [
                    ThoughtEntry(**thought) for thought in thoughts_data
                ]
            
            # Restore vibe_checks
            vibe_checks_file = session_dir / "vibe_checks.json"
            if vibe_checks_file.exists():
                with open(vibe_checks_file, 'r') as f:
                    vibe_checks_data = json.load(f)
                self.vibe_check_history[session_id] = [
                    VibeCheckEntry(**vibe_check) for vibe_check in vibe_checks_data
                ]
            
            logger.info(f"Restored session state for {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore session state for {session_id}: {e}")
            return False
    
    def get_reasoning_trace(self, session_id: str, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get reasoning trace for a session or specific agent.
        
        Args:
            session_id: Session identifier
            agent_id: Optional agent identifier to filter
            
        Returns:
            List of reasoning trace entries
        """
        thoughts = self.thought_history.get(session_id, [])
        
        if agent_id:
            thoughts = [thought for thought in thoughts if thought.agent_id == agent_id]
        
        # Convert to dictionary format
        trace = []
        for thought in thoughts:
            trace.append({
                'thought_number': thought.thought_number,
                'total_thoughts': thought.total_thoughts,
                'thought': thought.thought,
                'stage': thought.stage,
                'timestamp': thought.timestamp,
                'agent_id': thought.agent_id,
                'metadata': thought.metadata
            })
        
        return trace
    
    def get_self_assessment_history(self, session_id: str, 
                                  agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get self-assessment history for a session or specific agent.
        
        Args:
            session_id: Session identifier
            agent_id: Optional agent identifier to filter
            
        Returns:
            List of self-assessment entries
        """
        vibe_checks = self.vibe_check_history.get(session_id, [])
        
        if agent_id:
            vibe_checks = [vibe_check for vibe_check in vibe_checks if vibe_check.agent_id == agent_id]
        
        # Convert to dictionary format
        assessments = []
        for vibe_check in vibe_checks:
            assessments.append({
                'goal': vibe_check.goal,
                'plan': vibe_check.plan,
                'progress': vibe_check.progress,
                'uncertainties': vibe_check.uncertainties,
                'task_context': vibe_check.task_context,
                'timestamp': vibe_check.timestamp,
                'agent_id': vibe_check.agent_id,
                'assessment_score': vibe_check.assessment_score
            })
        
        return assessments
    
    def _summarize_thoughts(self, thoughts: List[ThoughtEntry]) -> Dict[str, Any]:
        """Summarize thoughts for cross-agent context."""
        if not thoughts:
            return {'count': 0, 'stages': [], 'recent_thoughts': []}
        
        stages = list(set(thought.stage for thought in thoughts))
        recent_thoughts = [
            {
                'agent_id': thought.agent_id,
                'stage': thought.stage,
                'thought': thought.thought[:200] + "..." if len(thought.thought) > 200 else thought.thought
            }
            for thought in sorted(thoughts, key=lambda x: x.timestamp, reverse=True)[:5]
        ]
        
        return {
            'count': len(thoughts),
            'stages': stages,
            'recent_thoughts': recent_thoughts
        }
    
    def _summarize_vibe_checks(self, vibe_checks: List[VibeCheckEntry]) -> Dict[str, Any]:
        """Summarize vibe_checks for cross-agent context."""
        if not vibe_checks:
            return {'count': 0, 'average_score': 0.0, 'recent_assessments': []}
        
        scores = [vibe_check.assessment_score for vibe_check in vibe_checks]
        recent_assessments = [
            {
                'agent_id': vibe_check.agent_id,
                'goal': vibe_check.goal,
                'assessment_score': vibe_check.assessment_score,
                'uncertainties_count': len(vibe_check.uncertainties)
            }
            for vibe_check in sorted(vibe_checks, key=lambda x: x.timestamp, reverse=True)[:3]
        ]
        
        return {
            'count': len(vibe_checks),
            'average_score': sum(scores) / len(scores) if scores else 0.0,
            'recent_assessments': recent_assessments
        }
    
    def _calculate_session_duration(self, session_id: str) -> float:
        """Calculate session duration in seconds."""
        thoughts = self.thought_history.get(session_id, [])
        if not thoughts:
            return 0.0
        
        timestamps = [thought.timestamp for thought in thoughts]
        return max(timestamps) - min(timestamps)
    
    def _get_agent_activity(self, session_id: str) -> Dict[str, Any]:
        """Get agent activity summary for session."""
        agent_activity = {}
        
        for context_key, agent_context in self.agent_contexts.items():
            if agent_context.session_id == session_id:
                agent_activity[agent_context.agent_id] = {
                    'role': agent_context.role,
                    'last_updated': agent_context.last_updated,
                    'task_count': len(agent_context.task_history),
                    'importance_score': agent_context.importance_score
                }
        
        return agent_activity
    
    def _load_persistent_data(self):
        """Load existing persistent data from disk."""
        try:
            for session_dir in self.persistence_dir.iterdir():
                if session_dir.is_dir():
                    session_id = session_dir.name
                    self.restore_session_state(session_id)
            
            logger.info(f"Loaded persistent data from {self.persistence_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to load persistent data: {e}")
    
    def cleanup_old_sessions(self, max_age_days: int = 30):
        """
        Clean up old session data.
        
        Args:
            max_age_days: Maximum age in days for session data
        """
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        
        # Clean up in-memory data
        for session_id in list(self.thought_history.keys()):
            thoughts = self.thought_history[session_id]
            if thoughts and max(thought.timestamp for thought in thoughts) < cutoff_time:
                del self.thought_history[session_id]
        
        for session_id in list(self.vibe_check_history.keys()):
            vibe_checks = self.vibe_check_history[session_id]
            if vibe_checks and max(vibe_check.timestamp for vibe_check in vibe_checks) < cutoff_time:
                del self.vibe_check_history[session_id]
        
        # Clean up disk data
        for session_dir in self.persistence_dir.iterdir():
            if session_dir.is_dir():
                session_id = session_dir.name
                session_state = self.session_states.get(session_id, {})
                last_updated = session_state.get('last_updated', 0)
                
                if last_updated < cutoff_time:
                    try:
                        import shutil
                        shutil.rmtree(session_dir)
                        logger.info(f"Cleaned up old session: {session_id}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up session {session_id}: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary and statistics."""
        total_sessions = len(self.session_states)
        total_thoughts = sum(len(thoughts) for thoughts in self.thought_history.values())
        total_vibe_checks = sum(len(vibe_checks) for vibe_checks in self.vibe_check_history.values())
        total_agents = len(self.agent_contexts)
        
        return {
            'total_sessions': total_sessions,
            'total_thoughts': total_thoughts,
            'total_vibe_checks': total_vibe_checks,
            'total_agents': total_agents,
            'persistence_dir': str(self.persistence_dir),
            'vector_db_stats': self.vector_db.get_statistics()
        }
