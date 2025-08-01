"""
Vector Memory module for storing and retrieving contextual information.
This is a stub implementation to support the PI orchestrator.
"""
from typing import List, Dict, Any, Optional


class VectorMemory:
    """
    A simple vector memory implementation for storing and retrieving contextual information.
    This is a stub implementation that uses in-memory storage.
    """
    
    def __init__(self):
        """Initialize the vector memory instance."""
        self._storage: List[Dict[str, Any]] = []
        self._next_id = 1
    
    def store_context(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store contextual information in the vector memory.
        
        Args:
            text: The text content to store
            metadata: Optional metadata associated with the content
            
        Returns:
            The ID of the stored content
        """
        entry = {
            'id': str(self._next_id),
            'text': text,
            'metadata': metadata or {},
            'timestamp': None  # Could add timestamp in real implementation
        }
        self._storage.append(entry)
        self._next_id += 1
        return entry['id']
    
    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant contextual information based on a query.
        
        Args:
            query: The search query
            top_k: Maximum number of results to return
            
        Returns:
            List of relevant context entries
        """
        # Simple implementation: return most recent entries that contain query terms
        results = []
        query_lower = query.lower()
        
        for entry in reversed(self._storage):  # Most recent first
            if query_lower in entry['text'].lower():
                results.append(entry)
                if len(results) >= top_k:
                    break
        
        return results
    
    def clear(self):
        """Clear all stored context."""
        self._storage.clear()
        self._next_id = 1
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all stored entries (for testing purposes)."""
        return self._storage.copy()