"""
Vector Database for AI Research Lab Memory Management

This module provides a vector database implementation for storing and retrieving
AI research content with semantic search capabilities. It uses FAISS for vector
operations and SQLite for metadata storage.
"""

import logging
import time
import json
import sqlite3
import threading
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np

# Try to import FAISS for vector operations
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Try to import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Thread-local storage for database connections
_thread_local = threading.local()

class VectorDatabase:
    """
    Vector database for storing and retrieving AI research content.
    
    This class provides semantic search capabilities for research content,
    including conversations, findings, and other text-based data. It uses
    FAISS for efficient vector operations and SQLite for metadata storage.
    """
    
    def __init__(self, db_path: str = "memory/vector_memory.db", 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 embedding_dim: int = 384):
        """
        Initialize the vector database.
        
        Args:
            db_path: Path to the SQLite database file
            embedding_model: Name of the sentence transformer model to use
            embedding_dim: Dimension of the embedding vectors
        """
        self.db_path = Path(db_path)
        self.embedding_model_name = embedding_model
        self.embedding_dim = embedding_dim
        
        # Ensure the database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._init_embedding_model()
        self._init_faiss_index()
        self._init_metadata_db()
        
        logger.info(f"VectorDatabase initialized with model: {embedding_model}")
    
    def _init_embedding_model(self):
        """Initialize the embedding model."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
            logger.warning("SentenceTransformers not available - using mock embeddings")
    
    def _init_faiss_index(self):
        """Initialize FAISS vector index."""
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index_to_id = {}  # Map FAISS index to our IDs
            logger.info(f"FAISS index initialized with dimension: {self.embedding_dim}")
        else:
            self.index = None
            self.mock_vectors = []  # Fallback storage
            logger.warning("Using mock vector storage - install faiss for better performance")
    
    def _get_db_connection(self):
        """Get thread-local database connection."""
        if not hasattr(_thread_local, 'vector_db'):
            _thread_local.vector_db = sqlite3.connect(str(self.db_path))
        return _thread_local.vector_db
    
    def _init_metadata_db(self):
        """Initialize SQLite database for metadata."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS vector_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_hash TEXT UNIQUE,
            content TEXT,
            content_type TEXT,
            agent_id TEXT,
            session_id TEXT,
            task_id TEXT,
            timestamp REAL,
            importance_score REAL DEFAULT 0.5,
            validated BOOLEAN DEFAULT FALSE,
            metadata TEXT,
            embedding_id INTEGER
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS context_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            summary_text TEXT,
            original_length INTEGER,
            compressed_length INTEGER,
            timestamp REAL,
            metadata TEXT
        )
        ''')
        
        conn.commit()
        logger.info("Metadata database initialized")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if self.embedding_model:
            return self.embedding_model.encode(text, convert_to_numpy=True)
        else:
            # Mock embedding for testing without dependencies
            import hashlib
            hash_value = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            np.random.seed(hash_value)
            return np.random.normal(0, 1, self.embedding_dim).astype(np.float32)
    
    def store_content(self, content: str, content_type: str = "conversation",
                     agent_id: Optional[str] = None, session_id: Optional[str] = None,
                     task_id: Optional[str] = None, importance_score: float = 0.5,
                     metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Store content with its embedding and metadata.
        
        Args:
            content: Text content to store
            content_type: Type of content (conversation, finding, etc.)
            agent_id: ID of agent that generated content
            session_id: Research session ID
            task_id: Task ID
            importance_score: Importance score (0.0-1.0)
            metadata: Additional metadata
            
        Returns:
            Content ID
        """
        # Generate embedding
        embedding = self.get_embedding(content)
        
        # Generate content hash for deduplication
        import hashlib
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Store in FAISS if available
            embedding_id = -1
            if self.index:
                self.index.add(embedding.reshape(1, -1))
                embedding_id = self.index.ntotal - 1
                self.index_to_id[embedding_id] = None  # Will be updated with DB ID
            
            # Store metadata in SQLite
            cursor.execute('''
            INSERT OR IGNORE INTO vector_metadata 
            (content_hash, content, content_type, agent_id, session_id, task_id, 
             timestamp, importance_score, metadata, embedding_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                content_hash, content, content_type, agent_id, session_id, task_id,
                time.time(), importance_score, json.dumps(metadata) if metadata else None, embedding_id
            ))
            
            # Get the inserted ID
            if cursor.rowcount > 0:
                content_id = cursor.lastrowid
                if self.index and embedding_id >= 0:
                    self.index_to_id[embedding_id] = content_id
            else:
                # Content already exists, get existing ID
                cursor.execute('SELECT id FROM vector_metadata WHERE content_hash = ?', (content_hash,))
                content_id = cursor.fetchone()[0]
            
            conn.commit()
            logger.debug(f"Stored content with ID: {content_id}")
            return content_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing content: {e}")
            raise
    
    def search_similar(self, query: str, limit: int = 5, 
                      content_type: Optional[str] = None,
                      session_id: Optional[str] = None,
                      min_importance: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar content using semantic similarity.
        
        Args:
            query: Search query
            limit: Maximum number of results
            content_type: Filter by content type
            session_id: Filter by session ID
            min_importance: Minimum importance score
            
        Returns:
            List of similar content items
        """
        # Generate query embedding
        query_embedding = self.get_embedding(query)
        
        if self.index and self.index.ntotal > 0:
            # Use FAISS for vector search
            query_embedding = query_embedding.reshape(1, -1)
            distances, indices = self.index.search(query_embedding, min(limit * 2, self.index.ntotal))
            
            # Get metadata for results
            results = []
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            for idx in indices[0]:
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                    
                content_id = self.index_to_id.get(idx)
                if content_id:
                    metadata = self._get_content_metadata(content_id)
                    if metadata and self._matches_filters(metadata, content_type, session_id, min_importance):
                        results.append(metadata)
                        if len(results) >= limit:
                            break
        else:
            # Fallback to mock search
            results = self._mock_search(query, limit, content_type, session_id, min_importance)
        
        return results
    
    def _matches_filters(self, metadata: Dict[str, Any], content_type: Optional[str],
                        session_id: Optional[str], min_importance: float) -> bool:
        """Check if metadata matches search filters."""
        if content_type and metadata.get('content_type') != content_type:
            return False
        if session_id and metadata.get('session_id') != session_id:
            return False
        if metadata.get('importance_score', 0) < min_importance:
            return False
        return True
    
    def _mock_search(self, query: str, limit: int, content_type: Optional[str],
                    session_id: Optional[str], min_importance: float) -> List[Dict[str, Any]]:
        """Mock search for testing without FAISS."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        # Build query with filters
        sql = '''
        SELECT id, content, content_type, agent_id, session_id, task_id, 
               timestamp, importance_score, metadata
        FROM vector_metadata
        WHERE importance_score >= ?
        '''
        params = [min_importance]
        
        if content_type:
            sql += ' AND content_type = ?'
            params.append(content_type)
        if session_id:
            sql += ' AND session_id = ?'
            params.append(session_id)
        
        sql += ' ORDER BY importance_score DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(sql, params)
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'content': row[1],
                'content_type': row[2],
                'agent_id': row[3],
                'session_id': row[4],
                'task_id': row[5],
                'timestamp': row[6],
                'importance_score': row[7],
                'metadata': json.loads(row[8]) if row[8] else {}
            })
        
        return results
    
    def _get_content_metadata(self, content_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific content ID."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, content, content_type, agent_id, session_id, task_id, 
               timestamp, importance_score, metadata
        FROM vector_metadata
        WHERE id = ?
        ''', (content_id,))
        
        row = cursor.fetchone()
        if row:
            return {
                'id': row[0],
                'content': row[1],
                'content_type': row[2],
                'agent_id': row[3],
                'session_id': row[4],
                'task_id': row[5],
                'timestamp': row[6],
                'importance_score': row[7],
                'metadata': json.loads(row[8]) if row[8] else {}
            }
        return None
    
    def update_importance(self, content_id: int, importance_score: float, 
                         validated: bool = False):
        """
        Update importance score and validation status for content.
        
        Args:
            content_id: ID of content to update
            importance_score: New importance score
            validated: Whether content has been validated
        """
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        UPDATE vector_metadata 
        SET importance_score = ?, validated = ?
        WHERE id = ?
        ''', (importance_score, validated, content_id))
        
        conn.commit()
        logger.debug(f"Updated importance for content {content_id}: {importance_score}")
    
    def store_context_summary(self, session_id: str, summary_text: str,
                            original_length: int, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Store a context summary for a session.
        
        Args:
            session_id: Session ID
            summary_text: Summary text
            original_length: Length of original content
            metadata: Additional metadata
            
        Returns:
            Summary ID
        """
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        compressed_length = len(summary_text)
        
        cursor.execute('''
        INSERT INTO context_summaries 
        (session_id, summary_text, original_length, compressed_length, timestamp, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            session_id, summary_text, original_length, compressed_length, 
            time.time(), json.dumps(metadata) if metadata else None
        ))
        
        summary_id = cursor.lastrowid
        conn.commit()
        
        # Also store summary in vector database for retrieval
        self.store_content(
            content=summary_text,
            content_type="summary",
            session_id=session_id,
            importance_score=0.8,  # Summaries are important
            metadata={'summary_id': summary_id, 'original_length': original_length}
        )
        
        logger.debug(f"Stored context summary with ID: {summary_id}")
        return summary_id
    
    def get_session_summaries(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all summaries for a session."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
        SELECT id, summary_text, original_length, compressed_length, timestamp, metadata
        FROM context_summaries
        WHERE session_id = ?
        ORDER BY timestamp
        ''', (session_id,))
        
        summaries = []
        for row in cursor.fetchall():
            summaries.append({
                'id': row[0],
                'summary_text': row[1],
                'original_length': row[2],
                'compressed_length': row[3],
                'timestamp': row[4],
                'metadata': json.loads(row[5]) if row[5] else {}
            })
        
        return summaries
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        # Count total content
        cursor.execute('SELECT COUNT(*) FROM vector_metadata')
        total_content = cursor.fetchone()[0]
        
        # Count by content type
        cursor.execute('''
        SELECT content_type, COUNT(*) 
        FROM vector_metadata 
        GROUP BY content_type
        ''')
        content_by_type = dict(cursor.fetchall())
        
        # Count validated content
        cursor.execute('SELECT COUNT(*) FROM vector_metadata WHERE validated = TRUE')
        validated_content = cursor.fetchone()[0]
        
        # Count summaries
        cursor.execute('SELECT COUNT(*) FROM context_summaries')
        total_summaries = cursor.fetchone()[0]
        
        # Vector index stats
        vector_count = self.index.ntotal if self.index else len(self.mock_vectors)
        
        return {
            'total_content': total_content,
            'content_by_type': content_by_type,
            'validated_content': validated_content,
            'total_summaries': total_summaries,
            'vector_count': vector_count,
            'embedding_dim': self.embedding_dim,
            'database_path': str(self.db_path)
        }
    
    def cleanup_old_content(self, days_old: int = 30, min_importance: float = 0.3):
        """
        Clean up old, low-importance content.
        
        Args:
            days_old: Remove content older than this many days
            min_importance: Keep content above this importance score
        """
        cutoff_time = time.time() - (days_old * 24 * 3600)
        
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
        DELETE FROM vector_metadata 
        WHERE timestamp < ? AND importance_score < ? AND validated = FALSE
        ''', (cutoff_time, min_importance))
        
        deleted_count = cursor.rowcount
        conn.commit()
        
        logger.info(f"Cleaned up {deleted_count} old content items")
        return deleted_count
    
    def close(self):
        """Close database connections."""
        if hasattr(_thread_local, 'vector_db'):
            _thread_local.vector_db.close()
            delattr(_thread_local, 'vector_db')
        logger.info("VectorDatabase closed")