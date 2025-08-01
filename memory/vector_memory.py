"""
Vector Memory implementation for storing and querying text embeddings.

This module provides a VectorMemory class that combines SQLite for metadata storage
and vector similarity search for text embeddings. It supports persistent storage
and efficient similarity-based text retrieval.
"""

import sqlite3
import json
import math
import os
from typing import List, Tuple, Optional
from collections import defaultdict


class VectorMemory:
    """
    A vector memory system for storing and querying text embeddings.
    
    Uses SQLite for persistent metadata storage and implements vector similarity
    search for finding similar texts. Embeddings are generated using a simple
    TF-IDF like approach for compatibility without external dependencies.
    """
    
    def __init__(self, db_path: str = "vector_memory.db"):
        """
        Initialize the VectorMemory instance.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize the SQLite database and create necessary tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS texts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization of text into words.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens (words)
        """
        # Simple tokenization - split on whitespace and punctuation
        import string
        # Remove punctuation and convert to lowercase
        translator = str.maketrans('', '', string.punctuation)
        cleaned = text.translate(translator).lower()
        return cleaned.split()
    
    def _compute_tf_idf_embedding(self, text: str, all_texts: List[str]) -> List[float]:
        """
        Compute a simple TF-IDF like embedding for the text.
        
        Args:
            text: The text to create embedding for
            all_texts: All texts in the corpus for IDF calculation
            
        Returns:
            Vector embedding as a list of floats
        """
        tokens = self._simple_tokenize(text)
        if not tokens:
            return [0.0] * 100  # Return zero vector for empty text
        
        # Build vocabulary from all texts (excluding current text if it's in the list)
        vocab = set()
        all_token_lists = []
        for t in all_texts:
            token_list = self._simple_tokenize(t)
            all_token_lists.append(token_list)
            vocab.update(token_list)
        
        if not vocab:
            return [0.0] * 100
        
        # Sort vocabulary for consistent ordering across embeddings
        vocab_list = sorted(vocab)
        vocab_size = min(len(vocab_list), 100)  # Limit vocabulary size
        vocab_list = vocab_list[:vocab_size]
        
        # Compute TF for current text
        tf = defaultdict(float)
        for token in tokens:
            if token in vocab_list:  # Only count tokens in our vocabulary
                tf[token] += 1.0
        
        # Normalize TF by document length
        total_tokens = len([t for t in tokens if t in vocab_list])
        if total_tokens > 0:
            for token in tf:
                tf[token] = tf[token] / total_tokens
        
        # Compute IDF
        idf = {}
        total_docs = len(all_token_lists)
        for token in vocab_list:
            doc_freq = sum(1 for token_list in all_token_lists if token in token_list)
            # Add smoothing to avoid division by zero and give reasonable IDF values
            idf[token] = math.log((total_docs + 1) / (doc_freq + 1)) + 1
        
        # Create TF-IDF vector with consistent ordering
        embedding = []
        for token in vocab_list:
            tf_val = tf.get(token, 0.0)
            idf_val = idf.get(token, 1.0)  # Default IDF of 1.0
            embedding.append(tf_val * idf_val)
        
        # Pad or truncate to exactly 100 dimensions
        while len(embedding) < 100:
            embedding.append(0.0)
        embedding = embedding[:100]
        
        # Normalize the vector (L2 normalization)
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]
        else:
            # If norm is 0, create a small random-like vector based on text hash
            text_hash = hash(text)
            embedding = [(text_hash % (i + 1)) / 1000.0 for i in range(100)]
            norm = math.sqrt(sum(x * x for x in embedding))
            if norm > 0:
                embedding = [x / norm for x in embedding]
        
        return embedding
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(a * a for a in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def add(self, text: str) -> None:
        """
        Store a text and its embedding in the vector memory.
        
        Args:
            text: The text to store
        """
        # Allow storing empty or whitespace-only texts
        pass
        
        # Get all existing texts for TF-IDF computation
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT text FROM texts")
            existing_texts = [row[0] for row in cursor.fetchall()]
        
        # Include the new text in the corpus for embedding computation
        all_texts = existing_texts + [text]
        
        # Compute embedding
        embedding = self._compute_tf_idf_embedding(text, all_texts)
        embedding_json = json.dumps(embedding)
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO texts (text, embedding) VALUES (?, ?)",
                (text, embedding_json)
            )
            conn.commit()
    
    def query(self, text: str, k: int = 5) -> List[str]:
        """
        Return the top-k most similar stored texts.
        
        Args:
            text: The query text
            k: Number of similar texts to return
            
        Returns:
            List of the most similar texts
        """
        if not text.strip():
            return []
        
        # Get all stored texts and embeddings
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT text, embedding FROM texts")
            stored_data = cursor.fetchall()
        
        if not stored_data:
            return []
        
        # Check for exact matches first
        exact_matches = []
        non_exact_data = []
        for stored_text, embedding_json in stored_data:
            if stored_text == text:
                exact_matches.append(stored_text)
            else:
                non_exact_data.append((stored_text, embedding_json))
        
        # If we have exact matches and they satisfy k, return them first
        if exact_matches and len(exact_matches) >= k:
            return exact_matches[:k]
        
        # Extract texts for TF-IDF computation
        stored_texts = [row[0] for row in stored_data]
        all_texts = stored_texts + [text]
        
        # Compute query embedding
        query_embedding = self._compute_tf_idf_embedding(text, all_texts)
        
        # Compute similarities for non-exact matches
        similarities = []
        for stored_text, embedding_json in non_exact_data:
            stored_embedding = json.loads(embedding_json)
            similarity = self._cosine_similarity(query_embedding, stored_embedding)
            similarities.append((similarity, stored_text))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Combine exact matches (highest priority) with similarity matches
        result = exact_matches[:]
        remaining_k = k - len(exact_matches)
        
        if remaining_k > 0:
            similar_texts = [text for _, text in similarities[:remaining_k]]
            result.extend(similar_texts)
        
        return result[:k]
    
    def __del__(self):
        """Cleanup when the object is destroyed."""
        # No explicit cleanup needed for SQLite connections as we use context managers
        pass