#!/usr/bin/env python3
"""
Simple test script for manuscript_drafter module.
"""

from manuscript_drafter import draft

def test_basic_functionality():
    """Test basic manuscript generation."""
    print("Testing basic functionality...")
    
    results = [
        {"description": "Sample A showed increased activity", "data": {"activity": 0.85}}
    ]
    context = {
        "objective": "investigate sample behavior",
        "methods": "Standard laboratory procedures were used"
    }
    
    output = draft(results, context)
    
    # Check that output is a string
    assert isinstance(output, str), "Output should be a string"
    print("✓ Output is a string")
    
    # Check for required sections
    assert "#" in output, "Should have a title"
    assert "## Abstract" in output, "Should have abstract section"
    assert "## Introduction" in output, "Should have introduction section"
    assert "## Methods" in output, "Should have methods section"
    assert "## Results" in output, "Should have results section"
    assert "## Discussion" in output, "Should have discussion section"
    assert "## Conclusion" in output, "Should have conclusion section"
    print("✓ All required sections present")
    
    print("Basic functionality test passed!")

def test_empty_input():
    """Test handling of empty input."""
    print("Testing empty input...")
    
    results = []
    context = {}
    
    output = draft(results, context)
    
    assert isinstance(output, str), "Output should be a string"
    assert len(output) > 0, "Output should not be empty"
    assert "Experimental Study" in output, "Should have default title"
    print("✓ Empty input handled correctly")

def test_error_handling():
    """Test error handling for invalid input types."""
    print("Testing error handling...")
    
    # Test with wrong types
    try:
        draft("not a list", {})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "results must be a list" in str(e), "Wrong error message"
        print("✓ Correctly caught invalid results type")
    
    try:
        draft([], "not a dict")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "context must be a dictionary" in str(e), "Wrong error message"
        print("✓ Correctly caught invalid context type")
    
    print("Error handling test passed!")

def test_multiple_results():
    """Test with multiple experimental results."""
    print("Testing multiple results...")
    
    results = [
        {"description": "First experiment", "data": {"value": 10}},
        {"description": "Second experiment", "data": {"value": 20}},
        {"finding": "Third experiment finding", "implication": "Important discovery"}
    ]
    context = {"study_type": "Comparative"}
    
    output = draft(results, context)
    
    assert "Result 1:" in output, "Should have Result 1"
    assert "Result 2:" in output, "Should have Result 2"
    assert "Result 3:" in output, "Should have Result 3"
    assert "First experiment" in output, "Should include first experiment"
    assert "Second experiment" in output, "Should include second experiment"
    assert "Third experiment finding" in output, "Should include third experiment"
    print("✓ Multiple results handled correctly")

def test_rich_context():
    """Test with comprehensive context information."""
    print("Testing rich context...")
    
    results = [{"outcome": "Successful experiment"}]
    context = {
        "study_type": "Longitudinal",
        "subject": "Plant Growth",
        "objective": "Study the effects of light on plant growth",
        "background": "Previous studies have shown varying light conditions affect plant development.",
        "significance": "Understanding light effects can improve agricultural practices.",
        "objectives": [
            "Measure growth rates under different light conditions",
            "Analyze chlorophyll content",
            "Compare root development"
        ],
        "methods": "Controlled environment experiments with standardized protocols.",
        "materials": ["Seeds", "Growth chambers", "Light meters"],
        "procedures": [
            "Prepare growth chambers",
            "Plant seeds in standardized soil",
            "Apply different light conditions",
            "Monitor growth daily"
        ],
        "conclusion": "Light intensity significantly affects plant growth patterns.",
        "interpretation": "The results confirm the importance of optimal lighting for plant development.",
        "limitations": [
            "Limited to laboratory conditions",
            "Short study duration",
            "Single plant species tested"
        ],
        "future_work": [
            "Extend to multiple species",
            "Field studies",
            "Long-term monitoring"
        ],
        "references": [
            {"authors": "Smith et al.", "year": "2020", "title": "Plant Growth Studies"},
            {"authors": "Johnson", "year": "2019", "title": "Light Effects on Plants"}
        ]
    }
    
    output = draft(results, context)
    
    # Check various sections
    assert "Longitudinal" in output, "Should include study type"
    assert "Plant Growth" in output, "Should include subject"
    assert "Study the effects of light on plant growth" in output, "Should include objective"
    assert "Previous studies have shown" in output, "Should include background"
    assert "Understanding light effects" in output, "Should include significance"
    assert "1. Measure growth rates" in output, "Should include objectives"
    assert "Controlled environment experiments" in output, "Should include methods"
    assert "* Seeds" in output, "Should include materials"
    assert "1. Prepare growth chambers" in output, "Should include procedures"
    assert "Light intensity significantly affects" in output, "Should include conclusion"
    assert "The results confirm" in output, "Should include interpretation"
    assert "* Limited to laboratory conditions" in output, "Should include limitations"
    assert "* Extend to multiple species" in output, "Should include future work"
    assert "1. Smith et al. (2020). Plant Growth Studies." in output, "Should include references"
    assert "2. Johnson (2019). Light Effects on Plants." in output, "Should include references"
    
    print("✓ Rich context handled correctly")

def main():
    """Run all tests."""
    print("Running manuscript drafter tests...\n")
    
    try:
        test_basic_functionality()
        test_empty_input()
        test_error_handling()
        test_multiple_results()
        test_rich_context()
        
        print("\n🎉 All tests passed!")
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)