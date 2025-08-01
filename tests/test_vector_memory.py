"""
Unit tests for VectorMemory class.

This module contains comprehensive tests for the VectorMemory implementation,
covering add/query functionality, persistence, and edge cases.
"""

import os
import tempfile
import pytest
from memory.vector_memory import VectorMemory


class TestVectorMemory:
    """Test suite for VectorMemory class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create temporary database file for testing
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_db_fd)  # Close the file descriptor
        self.vector_memory = VectorMemory(self.temp_db_path)
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Remove temporary database file
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
    
    def test_init_creates_database(self):
        """Test that initialization creates the database and tables."""
        # Database file should exist
        assert os.path.exists(self.temp_db_path)
        
        # Should be able to create another instance with same database
        vm2 = VectorMemory(self.temp_db_path)
        assert vm2 is not None
    
    def test_add_single_text(self):
        """Test adding a single text to the vector memory."""
        text = "This is a test document"
        self.vector_memory.add(text)
        
        # Should be able to query and get the same text back
        results = self.vector_memory.query(text, k=1)
        assert len(results) == 1
        assert results[0] == text
    
    def test_add_multiple_texts(self):
        """Test adding multiple texts to the vector memory."""
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Python is a powerful programming language",
            "Machine learning algorithms process data efficiently"
        ]
        
        for text in texts:
            self.vector_memory.add(text)
        
        # Query for each text should return it as the top result
        for text in texts:
            results = self.vector_memory.query(text, k=1)
            assert len(results) >= 1
            assert results[0] == text
    
    def test_query_similarity(self):
        """Test that query returns similar texts."""
        # Add some related texts
        self.vector_memory.add("Dogs are loyal pets")
        self.vector_memory.add("Cats are independent animals")
        self.vector_memory.add("Programming requires logical thinking")
        self.vector_memory.add("Puppies are young dogs")
        
        # Query with dog-related text
        results = self.vector_memory.query("Dogs are great companions", k=3)
        
        # Should return results
        assert len(results) <= 3
        assert len(results) > 0
        
        # The most similar should be about dogs
        assert any("Dogs" in result or "Puppies" in result for result in results[:2])
    
    def test_query_k_parameter(self):
        """Test that the k parameter limits the number of results."""
        texts = [
            "First text about cats",
            "Second text about dogs", 
            "Third text about birds",
            "Fourth text about fish",
            "Fifth text about rabbits"
        ]
        
        for text in texts:
            self.vector_memory.add(text)
        
        # Test different k values
        results_1 = self.vector_memory.query("Animals are interesting", k=1)
        results_3 = self.vector_memory.query("Animals are interesting", k=3)
        results_10 = self.vector_memory.query("Animals are interesting", k=10)
        
        assert len(results_1) == 1
        assert len(results_3) == 3
        assert len(results_10) == 5  # Should return all available texts
    
    def test_persistence_between_instances(self):
        """Test that data persists between different VectorMemory instances."""
        # Add text with first instance
        text1 = "Persistent text example"
        self.vector_memory.add(text1)
        
        # Create new instance with same database
        vm2 = VectorMemory(self.temp_db_path)
        
        # Should be able to query the text from the new instance
        results = vm2.query(text1, k=1)
        assert len(results) == 1
        assert results[0] == text1
        
        # Add text with second instance
        text2 = "Another persistent example"
        vm2.add(text2)
        
        # First instance should see both texts
        all_results = self.vector_memory.query("persistent example", k=5)
        assert len(all_results) == 2
        assert text1 in all_results
        assert text2 in all_results
    
    def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only texts."""
        # Adding empty text should not crash
        self.vector_memory.add("")
        self.vector_memory.add("   ")
        self.vector_memory.add("\t\n")
        
        # Add a real text
        real_text = "This is actual content"
        self.vector_memory.add(real_text)
        
        # Query should still work
        results = self.vector_memory.query(real_text, k=5)
        assert len(results) == 1
        assert results[0] == real_text
    
    def test_empty_query(self):
        """Test querying with empty text."""
        # Add some texts
        self.vector_memory.add("Sample text for testing")
        
        # Empty query should return empty list
        results = self.vector_memory.query("", k=5)
        assert results == []
        
        results = self.vector_memory.query("   ", k=5)
        assert results == []
    
    def test_query_no_data(self):
        """Test querying when no texts are stored."""
        results = self.vector_memory.query("Query with no data", k=5)
        assert results == []
    
    def test_special_characters(self):
        """Test handling of texts with special characters."""
        texts = [
            "Text with punctuation: Hello, world!",
            "Numbers 123 and symbols @#$%",
            "Unicode: café, naïve, résumé",
            "Newlines\nand\ttabs"
        ]
        
        for text in texts:
            self.vector_memory.add(text)
        
        # Should be able to query and retrieve
        for text in texts:
            results = self.vector_memory.query(text, k=1)
            assert len(results) == 1
            assert results[0] == text
    
    def test_large_text(self):
        """Test handling of large text inputs."""
        # Create a large text
        large_text = "Large text content. " * 1000
        
        self.vector_memory.add(large_text)
        
        # Should be able to query and retrieve
        results = self.vector_memory.query(large_text[:100], k=1)
        assert len(results) == 1
        assert results[0] == large_text
    
    def test_case_sensitivity(self):
        """Test case sensitivity in text matching."""
        self.vector_memory.add("Python Programming Language")
        self.vector_memory.add("python programming language")
        
        # Both versions should be stored and retrievable
        results = self.vector_memory.query("python programming", k=5)
        assert len(results) == 2
    
    def test_duplicate_texts(self):
        """Test adding duplicate texts."""
        text = "Duplicate text example"
        
        # Add the same text multiple times
        self.vector_memory.add(text)
        self.vector_memory.add(text)
        self.vector_memory.add(text)
        
        # Should be able to retrieve all instances
        results = self.vector_memory.query(text, k=5)
        assert len(results) == 3
        assert all(result == text for result in results)
    
    def test_default_database_path(self):
        """Test VectorMemory with default database path."""
        # Clean up any existing default database
        default_db = "vector_memory.db"
        if os.path.exists(default_db):
            os.unlink(default_db)
        
        try:
            # Create VectorMemory with default path
            vm = VectorMemory()
            vm.add("Test with default database")
            
            results = vm.query("Test with default", k=1)
            assert len(results) == 1
            assert "Test with default database" in results[0]
            
        finally:
            # Clean up
            if os.path.exists(default_db):
                os.unlink(default_db)
    
    def test_concurrent_access(self):
        """Test that multiple VectorMemory instances can access the same database."""
        vm1 = VectorMemory(self.temp_db_path)
        vm2 = VectorMemory(self.temp_db_path)
        
        # Add text with first instance
        vm1.add("Text from instance 1")
        
        # Add text with second instance  
        vm2.add("Text from instance 2")
        
        # Both instances should see both texts
        results1 = vm1.query("Text from", k=5)
        results2 = vm2.query("Text from", k=5)
        
        assert len(results1) == 2
        assert len(results2) == 2
        assert set(results1) == set(results2)