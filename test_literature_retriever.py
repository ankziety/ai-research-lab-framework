#!/usr/bin/env python3
"""
Test script for the Literature Retriever with free APIs.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from literature_retriever import LiteratureRetriever

def test_free_apis():
    """Test the literature retriever with free APIs."""
    print("Testing Literature Retriever with free APIs...")
    
    # Initialize retriever
    retriever = LiteratureRetriever()
    
    # Test search with free APIs
    query = "machine learning interpretability"
    print(f"\nSearching for: '{query}'")
    
    try:
        # Test with default sources (free APIs)
        results = retriever.search(query, max_results=10)
        
        print(f"Found {len(results)} papers")
        
        if results:
            print("\nSample results:")
            for i, paper in enumerate(results[:3]):
                print(f"\n{i+1}. {paper.get('title', 'No title')}")
                print(f"   Authors: {', '.join(paper.get('authors', ['Unknown']))}")
                print(f"   Year: {paper.get('publication_year', 'Unknown')}")
                print(f"   Source: {paper.get('source', 'Unknown')}")
                print(f"   URL: {paper.get('url', 'No URL')}")
        
        # Test API key status
        status = retriever.get_api_key_status()
        print(f"\nAPI Key Status:")
        print(f"  Valid keys: {status.get('valid_keys', [])}")
        print(f"  Missing keys: {status.get('missing_keys', [])}")
        print(f"  Warnings: {status.get('warnings', [])}")
        
        return True
        
    except Exception as e:
        print(f"Error testing literature retriever: {e}")
        return False

def test_specific_sources():
    """Test specific free sources."""
    print("\nTesting specific free sources...")
    
    retriever = LiteratureRetriever()
    query = "artificial intelligence ethics"
    
    sources_to_test = ['pubmed', 'arxiv', 'crossref', 'semantic_scholar']
    
    for source in sources_to_test:
        try:
            print(f"\nTesting {source}...")
            results = retriever.search(query, max_results=5, sources=[source])
            print(f"  Found {len(results)} papers from {source}")
            
            if results:
                paper = results[0]
                print(f"  Sample: {paper.get('title', 'No title')[:60]}...")
                
        except Exception as e:
            print(f"  Error with {source}: {e}")

if __name__ == "__main__":
    print("Literature Retriever Test with Free APIs")
    print("=" * 50)
    
    success = test_free_apis()
    test_specific_sources()
    
    if success:
        print("\n✅ Literature retriever tests completed successfully!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)