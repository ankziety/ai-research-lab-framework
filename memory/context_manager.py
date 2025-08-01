"""
Context Manager for handling conversation context overflow and retrieval.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from .vector_database import VectorDatabase

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages conversation context, handles overflow situations,
    and provides intelligent context retrieval.
    """
    
    def __init__(self, vector_db: VectorDatabase, max_context_length: int = 4000):
        """
        Initialize context manager.
        
        Args:
            vector_db: VectorDatabase instance for storage/retrieval
            max_context_length: Maximum context length before overflow handling
        """
        self.vector_db = vector_db
        self.max_context_length = max_context_length
        self.active_contexts = {}  # session_id -> context data
        
        logger.info(f"ContextManager initialized with max length: {max_context_length}")
    
    def add_to_context(self, session_id: str, content: str, 
                      content_type: str = "conversation",
                      agent_id: Optional[str] = None,
                      importance_score: float = 0.5,
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add content to session context, handling overflow if necessary.
        
        Args:
            session_id: Session identifier
            content: Content to add
            content_type: Type of content
            agent_id: ID of agent that generated content
            importance_score: Importance score for the content
            metadata: Additional metadata
            
        Returns:
            True if content was added, False if overflow occurred
        """
        # Initialize session context if needed
        if session_id not in self.active_contexts:
            self.active_contexts[session_id] = {
                'content_items': [],
                'total_length': 0,
                'last_summary_time': time.time(),
                'overflow_count': 0
            }
        
        context = self.active_contexts[session_id]
        content_length = len(content)
        
        # Check if adding this content would cause overflow
        if context['total_length'] + content_length > self.max_context_length:
            logger.info(f"Context overflow detected for session {session_id}")
            overflow_handled = self._handle_context_overflow(session_id)
            if not overflow_handled:
                logger.warning(f"Failed to handle overflow for session {session_id}")
                return False
        
        # Add content to context
        content_item = {
            'content': content,
            'content_type': content_type,
            'agent_id': agent_id,
            'timestamp': time.time(),
            'importance_score': importance_score,
            'metadata': metadata or {},
            'length': content_length
        }
        
        context['content_items'].append(content_item)
        context['total_length'] += content_length
        
        # Store in vector database for long-term retrieval
        self.vector_db.store_content(
            content=content,
            content_type=content_type,
            agent_id=agent_id,
            session_id=session_id,
            importance_score=importance_score,
            metadata=metadata
        )
        
        logger.debug(f"Added content to session {session_id}: {content_length} chars")
        return True
    
    def _handle_context_overflow(self, session_id: str) -> bool:
        """
        Handle context overflow by summarizing and compressing old content.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if overflow was successfully handled
        """
        context = self.active_contexts[session_id]
        content_items = context['content_items']
        
        if not content_items:
            return True
        
        # Determine how much content to summarize (oldest 50%)
        total_items = len(content_items)
        items_to_summarize = max(1, total_items // 2)
        
        # Separate content to keep vs summarize
        items_to_summarize = content_items[:items_to_summarize]
        items_to_keep = content_items[items_to_summarize:]
        
        # Create summary of items to be compressed
        summary_text = self._create_context_summary(items_to_summarize, session_id)
        original_length = sum(item['length'] for item in items_to_summarize)
        
        # Store summary in vector database
        summary_id = self.vector_db.store_context_summary(
            session_id=session_id,
            summary_text=summary_text,
            original_length=original_length,
            metadata={
                'items_summarized': len(items_to_summarize),
                'compression_ratio': len(summary_text) / max(1, original_length)
            }
        )
        
        # Update context to keep only recent items + summary
        summary_item = {
            'content': summary_text,
            'content_type': 'summary',
            'agent_id': 'context_manager',
            'timestamp': time.time(),
            'importance_score': 0.8,
            'metadata': {'summary_id': summary_id, 'items_summarized': len(items_to_summarize)},
            'length': len(summary_text)
        }
        
        context['content_items'] = [summary_item] + items_to_keep
        context['total_length'] = sum(item['length'] for item in context['content_items'])
        context['last_summary_time'] = time.time()
        context['overflow_count'] += 1
        
        logger.info(f"Context overflow handled for session {session_id}: "
                   f"summarized {len(items_to_summarize)} items, "
                   f"compressed {original_length} to {len(summary_text)} chars")
        
        return True
    
    def _create_context_summary(self, items: List[Dict[str, Any]], session_id: str) -> str:
        """
        Create a summary of context items.
        
        This is a simplified implementation. In a real system, this would use
        an LLM to create intelligent summaries.
        
        Args:
            items: List of content items to summarize
            session_id: Session identifier
            
        Returns:
            Summary text
        """
        # Group items by agent and content type for better summarization
        agent_contributions = {}
        for item in items:
            agent_id = item.get('agent_id', 'unknown')
            if agent_id not in agent_contributions:
                agent_contributions[agent_id] = []
            agent_contributions[agent_id].append(item)
        
        summary_parts = [f"=== Context Summary for Session {session_id} ==="]
        
        for agent_id, contributions in agent_contributions.items():
            if not contributions:
                continue
                
            summary_parts.append(f"\n{agent_id.upper()} Contributions:")
            
            # Summarize by content type
            by_type = {}
            for item in contributions:
                content_type = item['content_type']
                if content_type not in by_type:
                    by_type[content_type] = []
                by_type[content_type].append(item['content'])
            
            for content_type, contents in by_type.items():
                if content_type == 'conversation':
                    # Extract key points from conversations
                    key_points = self._extract_key_points(contents)
                    summary_parts.append(f"- {content_type.title()}: {'; '.join(key_points)}")
                else:
                    # For other types, provide count and brief description
                    summary_parts.append(f"- {content_type.title()}: {len(contents)} items")
        
        summary_parts.append(f"\n[Summarized {len(items)} items from context]")
        
        return "\n".join(summary_parts)
    
    def _extract_key_points(self, conversations: List[str]) -> List[str]:
        """
        Extract key points from conversation texts.
        Simplified implementation - real version would use NLP.
        
        Args:
            conversations: List of conversation texts
            
        Returns:
            List of key points
        """
        all_text = " ".join(conversations).lower()
        
        # Intelligent keyword extraction
        important_keywords = [
            'research', 'analysis', 'finding', 'result', 'conclusion',
            'recommendation', 'hypothesis', 'data', 'evidence', 'study'
        ]
        
        key_points = []
        for keyword in important_keywords:
            if keyword in all_text:
                # Find sentence containing keyword
                sentences = [s.strip() for s in " ".join(conversations).split('.') if s.strip()]
                for sentence in sentences:
                    if keyword in sentence.lower() and len(sentence) < 150:
                        key_points.append(sentence.strip())
                        break
        
        return key_points[:config.get('max_key_points', 10)]  # Configurable limit
    
    def get_current_context(self, session_id: str) -> Optional[str]:
        """
        Get current context for a session as formatted string.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Formatted context string or None if session not found
        """
        if session_id not in self.active_contexts:
            return None
        
        context = self.active_contexts[session_id]
        context_parts = []
        
        for item in context['content_items']:
            timestamp_str = time.strftime('%H:%M:%S', time.localtime(item['timestamp']))
            agent_prefix = f"[{item.get('agent_id', 'unknown')}@{timestamp_str}]"
            context_parts.append(f"{agent_prefix} {item['content']}")
        
        return "\n".join(context_parts)
    
    def retrieve_relevant_context(self, session_id: str, query: str, 
                                max_items: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from vector database based on query.
        
        Args:
            session_id: Session identifier
            query: Query to search for relevant context
            max_items: Maximum number of context items to retrieve
            
        Returns:
            List of relevant context items with similarity scores
        """
        logger.debug(f"Retrieving relevant context for session {session_id}: {query}")
        
        # Search vector database for relevant content
        results = self.vector_db.search_similar(
            query=query,
            limit=max_items,
            session_id=session_id,
            min_importance=0.3  # Filter out low-importance content
        )
        
        return results
    
    def inject_relevant_context(self, session_id: str, current_prompt: str,
                              max_context_items: int = 3) -> str:
        """
        Inject relevant historical context into current prompt.
        
        Args:
            session_id: Session identifier
            current_prompt: Current prompt/question
            max_context_items: Maximum context items to inject
            
        Returns:
            Enhanced prompt with relevant context
        """
        # Get relevant context
        relevant_context = self.retrieve_relevant_context(
            session_id=session_id,
            query=current_prompt,
            max_items=max_context_items
        )
        
        if not relevant_context:
            return current_prompt
        
        # Format context for injection
        context_strings = []
        for item in relevant_context:
            timestamp_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(item['timestamp']))
            context_strings.append(
                f"[{timestamp_str}] {item['content'][:200]}{'...' if len(item['content']) > 200 else ''}"
            )
        
        # Create enhanced prompt
        enhanced_prompt = f"""
{current_prompt}

--- Relevant Context ---
{chr(10).join(context_strings)}
--- End Context ---

Please consider the above context when formulating your response.
"""
        
        logger.debug(f"Injected {len(relevant_context)} context items into prompt")
        return enhanced_prompt
    
    def get_context_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about context usage.
        
        Args:
            session_id: Optional specific session ID
            
        Returns:
            Context statistics
        """
        if session_id:
            if session_id not in self.active_contexts:
                return {'error': 'Session not found'}
            
            context = self.active_contexts[session_id]
            return {
                'session_id': session_id,
                'total_items': len(context['content_items']),
                'total_length': context['total_length'],
                'overflow_count': context['overflow_count'],
                'last_summary_time': context['last_summary_time'],
                'max_context_length': self.max_context_length,
                'utilization': context['total_length'] / self.max_context_length
            }
        else:
            # Global statistics
            total_sessions = len(self.active_contexts)
            total_items = sum(len(ctx['content_items']) for ctx in self.active_contexts.values())
            total_length = sum(ctx['total_length'] for ctx in self.active_contexts.values())
            total_overflows = sum(ctx['overflow_count'] for ctx in self.active_contexts.values())
            
            return {
                'total_sessions': total_sessions,
                'total_context_items': total_items,
                'total_context_length': total_length,
                'total_overflows': total_overflows,
                'max_context_length': self.max_context_length,
                'average_utilization': (total_length / max(1, total_sessions)) / self.max_context_length
            }
    
    def clear_session_context(self, session_id: str):
        """
        Clear context for a specific session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.active_contexts:
            del self.active_contexts[session_id]
            logger.info(f"Cleared context for session: {session_id}")
    
    def cleanup_old_sessions(self, hours_old: int = 24):
        """
        Clean up contexts for old inactive sessions.
        
        Args:
            hours_old: Remove sessions older than this many hours
        """
        cutoff_time = time.time() - (hours_old * 3600)
        
        sessions_to_remove = []
        for session_id, context in self.active_contexts.items():
            if context['last_summary_time'] < cutoff_time:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.active_contexts[session_id]
        
        logger.info(f"Cleaned up {len(sessions_to_remove)} old session contexts")
        return len(sessions_to_remove)