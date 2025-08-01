#!/usr/bin/env python3
"""
Demonstration script for the PI Orchestrator module.

This script shows how to use the PI orchestrator to register specialists,
set up memory, and run research tasks.
"""
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pi_orchestrator import PIOrchestrator
from vector_memory import VectorMemory


def mock_literature_reviewer(request: str) -> dict:
    """Mock literature reviewer specialist."""
    return {
        "summary": f"Literature review completed for: {request}",
        "papers_found": 12,
        "key_findings": [
            "Recent advances in transformer architectures",
            "Improved performance on benchmark datasets",
            "Novel attention mechanisms"
        ],
        "recommendations": "Focus on attention-based models for next phase"
    }


def mock_data_analyst(request: str) -> dict:
    """Mock data analyst specialist."""
    return {
        "analysis": f"Data analysis completed for: {request}",
        "statistics": {
            "samples": 1000,
            "mean_accuracy": 0.87,
            "std_deviation": 0.12
        },
        "insights": [
            "Model performance varies significantly across datasets",
            "Larger models show diminishing returns",
            "Data quality is the primary factor in performance"
        ],
        "visualization_paths": ["/tmp/plot1.png", "/tmp/plot2.png"]
    }


def mock_manuscript_writer(request: str) -> dict:
    """Mock manuscript writer specialist."""
    return {
        "draft": f"Manuscript draft created for: {request}",
        "sections": ["Abstract", "Introduction", "Methods", "Results", "Discussion"],
        "word_count": 4500,
        "references": 25,
        "status": "First draft complete, ready for review"
    }


def main():
    """Demonstrate PI orchestrator functionality."""
    print("=== PI Orchestrator Demonstration ===\n")
    
    # Initialize the orchestrator and memory
    orchestrator = PIOrchestrator()
    memory = VectorMemory()
    
    print("1. Setting up vector memory...")
    orchestrator.set_memory(memory)
    
    print("2. Registering specialist agents...")
    orchestrator.register_specialist("literature_reviewer", mock_literature_reviewer)
    orchestrator.register_specialist("data_analyst", mock_data_analyst) 
    orchestrator.register_specialist("manuscript_writer", mock_manuscript_writer)
    
    print(f"   Registered specialists: {orchestrator.get_registered_specialists()}")
    print()
    
    # Test different types of research requests
    test_requests = [
        "Review recent literature on transformer models and analyze performance data",
        "Analyze experimental results from our machine learning study",
        "Write a manuscript draft on deep learning applications",
        "What are the latest trends in artificial intelligence?"
    ]
    
    for i, request in enumerate(test_requests, 1):
        print(f"=== Test {i}: Running Research Task ===")
        print(f"Request: {request}")
        print()
        
        result = orchestrator.run_research_task(request)
        
        print(f"Task ID: {result['task_id']}")
        print(f"Status: {result['status']}")
        print(f"Specialists involved: {result['specialist_count']}")
        
        if result.get('results'):
            print("Results by specialist:")
            for specialist, output in result['results'].items():
                print(f"  - {specialist}: {list(output.keys()) if isinstance(output, dict) else output}")
        
        if result.get('errors'):
            print("Errors encountered:")
            for specialist, error in result['errors'].items():
                print(f"  - {specialist}: {error}")
        
        print()
    
    # Show provenance logging
    print("=== Provenance Log ===")
    log = orchestrator.get_provenance_log()
    print(f"Total log entries: {len(log)}")
    
    # Show recent actions
    print("Recent actions:")
    for entry in log[-5:]:  # Last 5 entries
        print(f"  - {entry['action']} at {entry['timestamp']}")
    
    print()
    
    # Show memory contents
    print("=== Vector Memory Contents ===")
    stored_entries = memory.get_all()
    print(f"Total stored entries: {len(stored_entries)}")
    
    # Test memory retrieval
    print("Testing memory retrieval for 'literature':")
    relevant_entries = memory.retrieve_context("literature", top_k=3)
    for entry in relevant_entries:
        print(f"  - {entry['text'][:100]}...")
    
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()