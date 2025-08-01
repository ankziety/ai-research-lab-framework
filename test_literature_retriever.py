"""
Unit tests for literature_retriever.py

Tests the Literature Retriever module functionality including:
- Search method with various inputs
- Error handling and edge cases
- Mock API responses
- API URL construction and response parsing structure
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import sys
import os

# Add the parent directory to the path to import literature_retriever
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from literature_retriever import LiteratureRetriever, search_literature


class TestLiteratureRetriever(unittest.TestCase):
    """Test cases for the LiteratureRetriever class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.retriever = LiteratureRetriever()
        self.sample_query = "machine learning"
        
    def test_init_default_values(self):
        """Test LiteratureRetriever initialization with default values."""
        retriever = LiteratureRetriever()
        self.assertEqual(retriever.api_base_url, "https://api.pubmed.ncbi.nlm.nih.gov")
        self.assertIsNone(retriever.api_key)
        self.assertEqual(retriever.session_timeout, 30)
        
    def test_init_custom_values(self):
        """Test LiteratureRetriever initialization with custom values."""
        custom_url = "https://custom.api.url"
        custom_key = "test_api_key"
        retriever = LiteratureRetriever(api_base_url=custom_url, api_key=custom_key)
        self.assertEqual(retriever.api_base_url, custom_url)
        self.assertEqual(retriever.api_key, custom_key)
        
    def test_search_basic_functionality(self):
        """Test basic search functionality."""
        results = self.retriever.search(self.sample_query)
        
        # Should return a list
        self.assertIsInstance(results, list)
        
        # Should return default 5 results
        self.assertEqual(len(results), 5)
        
        # Each result should be a dictionary with required fields
        for result in results:
            self.assertIsInstance(result, dict)
            self.assertIn("title", result)
            self.assertIn("authors", result)
            self.assertIn("abstract", result)
            self.assertIn("publication_date", result)
            self.assertIn("url", result)
            self.assertIn("source", result)
            
    def test_search_custom_max_results(self):
        """Test search with custom max_results parameter."""
        max_results = 3
        results = self.retriever.search(self.sample_query, max_results=max_results)
        
        self.assertEqual(len(results), max_results)
        
    def test_search_max_results_larger_than_available(self):
        """Test search when requesting more results than available."""
        # The mock data has 5 papers, request 10
        max_results = 10
        results = self.retriever.search(self.sample_query, max_results=max_results)
        
        # Should return only available results (5)
        self.assertEqual(len(results), 5)
        
    def test_search_empty_query_raises_error(self):
        """Test that empty query raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.retriever.search("")
        self.assertIn("Query cannot be empty", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            self.retriever.search("   ")  # Whitespace only
        self.assertIn("Query cannot be empty", str(context.exception))
        
    def test_search_invalid_max_results_raises_error(self):
        """Test that invalid max_results raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.retriever.search(self.sample_query, max_results=0)
        self.assertIn("max_results must be greater than 0", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            self.retriever.search(self.sample_query, max_results=-1)
        self.assertIn("max_results must be greater than 0", str(context.exception))
        
    def test_search_result_structure(self):
        """Test that search results have the expected structure."""
        results = self.retriever.search(self.sample_query, max_results=1)
        result = results[0]
        
        # Test required fields exist
        required_fields = ["title", "authors", "abstract", "publication_date", "url", "source"]
        for field in required_fields:
            self.assertIn(field, result)
            
        # Test field types
        self.assertIsInstance(result["title"], str)
        self.assertIsInstance(result["authors"], list)
        self.assertIsInstance(result["abstract"], str)
        self.assertIsInstance(result["publication_date"], str)
        self.assertIsInstance(result["url"], str)
        self.assertIsInstance(result["source"], str)
        
        # Test that title contains the query term
        self.assertIn(self.sample_query, result["title"])
        
    def test_search_different_sources(self):
        """Test that search results include different sources."""
        results = self.retriever.search(self.sample_query, max_results=5)
        
        sources = [result["source"] for result in results]
        
        # Should have both PubMed and ArXiv sources
        self.assertIn("PubMed", sources)
        self.assertIn("ArXiv", sources)
        
    def test_construct_api_url_basic(self):
        """Test API URL construction with basic parameters."""
        url = self.retriever._construct_api_url("test query", 10)
        
        self.assertIn("esearch.fcgi", url)
        self.assertIn("db=pubmed", url)
        self.assertIn("term=test%20query", url)  # URL encoded
        self.assertIn("retmax=10", url)
        self.assertIn("retmode=json", url)
        
    def test_construct_api_url_with_api_key(self):
        """Test API URL construction with API key."""
        retriever = LiteratureRetriever(api_key="test_key")
        url = retriever._construct_api_url("test", 5)
        
        self.assertIn("api_key=test_key", url)
        
    def test_parse_pubmed_response_structure(self):
        """Test PubMed response parsing structure."""
        # Test with empty response
        empty_response = {}
        papers = self.retriever._parse_pubmed_response(empty_response)
        self.assertEqual(papers, [])
        
        # Test with mock response structure
        mock_response = {
            "esearchresult": {
                "idlist": ["12345", "67890"]
            }
        }
        papers = self.retriever._parse_pubmed_response(mock_response)
        # Currently returns empty list as it's a stub
        self.assertEqual(papers, [])
        
    def test_parse_arxiv_response_structure(self):
        """Test ArXiv response parsing structure."""
        mock_xml = "<feed></feed>"
        papers = self.retriever._parse_arxiv_response(mock_xml)
        # Currently returns empty list as it's a stub
        self.assertEqual(papers, [])
        
    @patch('literature_retriever.time.sleep')
    def test_stubbed_search_timing(self, mock_sleep):
        """Test that stubbed search includes simulated delay."""
        self.retriever._stubbed_search("test", 1)
        mock_sleep.assert_called_once_with(0.1)
        
    def test_search_query_in_results(self):
        """Test that search query appears in relevant result fields."""
        query = "neural networks"
        results = self.retriever.search(query, max_results=2)
        
        for result in results:
            # Query should appear in title or keywords
            query_found = (query in result["title"] or 
                          any(query.lower() in keyword for keyword in result.get("keywords", [])))
            self.assertTrue(query_found, f"Query '{query}' not found in result: {result['title']}")
            
    @patch('literature_retriever.LiteratureRetriever._stubbed_search')
    def test_search_exception_handling(self, mock_stubbed_search):
        """Test that search method handles exceptions properly."""
        mock_stubbed_search.side_effect = Exception("API Error")
        
        with self.assertRaises(RuntimeError) as context:
            self.retriever.search("test query")
            
        self.assertIn("Failed to retrieve literature", str(context.exception))
        self.assertIn("API Error", str(context.exception))


class TestConvenienceFunction(unittest.TestCase):
    """Test cases for the convenience function."""
    
    def test_search_literature_function(self):
        """Test the search_literature convenience function."""
        results = search_literature("artificial intelligence", max_results=3)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        
        # Should have same structure as class method
        for result in results:
            self.assertIn("title", result)
            self.assertIn("authors", result)
            self.assertIn("abstract", result)
            
    def test_search_literature_default_parameters(self):
        """Test convenience function with default parameters."""
        results = search_literature("test")
        
        # Should use default max_results=5
        self.assertEqual(len(results), 5)


class TestMockAPIResponses(unittest.TestCase):
    """Test cases for mock API response handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.retriever = LiteratureRetriever()
        
    def test_mock_response_diversity(self):
        """Test that mock responses provide diverse paper types."""
        results = self.retriever.search("deep learning", max_results=5)
        
        # Should have different types of publications
        journals = [result["journal"] for result in results]
        self.assertTrue(len(set(journals)) > 1, "Should have diverse journal sources")
        
        # Should have both papers with and without DOIs
        dois = [result.get("doi") for result in results]
        self.assertIn(None, dois, "Should have some papers without DOI (ArXiv)")
        self.assertTrue(any(doi for doi in dois), "Should have some papers with DOI")
        
    def test_mock_response_identifiers(self):
        """Test that mock responses have proper identifiers."""
        results = self.retriever.search("test", max_results=5)
        
        pubmed_papers = [r for r in results if r["source"] == "PubMed"]
        arxiv_papers = [r for r in results if r["source"] == "ArXiv"]
        
        # PubMed papers should have PMIDs
        for paper in pubmed_papers:
            self.assertIn("pmid", paper)
            self.assertIsInstance(paper["pmid"], str)
            
        # ArXiv papers should have ArXiv IDs
        for paper in arxiv_papers:
            self.assertIn("arxiv_id", paper)
            self.assertIsInstance(paper["arxiv_id"], str)
            self.assertTrue(paper["arxiv_id"].startswith("2"), "ArXiv ID should start with year")


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)