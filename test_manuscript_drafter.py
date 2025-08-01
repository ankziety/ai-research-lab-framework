"""
Unit tests for manuscript_drafter module.

Tests the functionality of generating structured Markdown drafts
from experimental results and contextual information.
"""

import pytest
from manuscript_drafter import draft


class TestManuscriptDrafter:
    """Test cases for the manuscript drafter functionality."""
    
    def test_basic_functionality(self):
        """Test basic manuscript generation with minimal input."""
        results = [
            {"description": "Sample A showed increased activity", "data": {"activity": 0.85}}
        ]
        context = {
            "objective": "investigate sample behavior",
            "methods": "Standard laboratory procedures were used"
        }
        
        output = draft(results, context)
        
        # Check that output is a string
        assert isinstance(output, str)
        
        # Check for required sections
        assert "#" in output  # Title
        assert "## Abstract" in output
        assert "## Introduction" in output
        assert "## Methods" in output
        assert "## Results" in output
        assert "## Discussion" in output
        assert "## Conclusion" in output
    
    def test_empty_input(self):
        """Test handling of empty input."""
        results = []
        context = {}
        
        output = draft(results, context)
        
        assert isinstance(output, str)
        assert len(output) > 0
        assert "Experimental Study" in output  # Default title
    
    def test_minimal_results(self):
        """Test with minimal results data."""
        results = [{"outcome": "Positive result"}]
        context = {}
        
        output = draft(results, context)
        
        assert "Result 1:" in output
        assert "Positive result" in output
    
    def test_multiple_results(self):
        """Test with multiple experimental results."""
        results = [
            {"description": "First experiment", "data": {"value": 10}},
            {"description": "Second experiment", "data": {"value": 20}},
            {"finding": "Third experiment finding", "implication": "Important discovery"}
        ]
        context = {"study_type": "Comparative"}
        
        output = draft(results, context)
        
        assert "Result 1:" in output
        assert "Result 2:" in output
        assert "Result 3:" in output
        assert "First experiment" in output
        assert "Second experiment" in output
        assert "Third experiment finding" in output
    
    def test_rich_context(self):
        """Test with comprehensive context information."""
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
        
        # Check title generation
        assert "Longitudinal" in output
        assert "Plant Growth" in output
        
        # Check abstract
        assert "Study the effects of light on plant growth" in output
        
        # Check introduction
        assert "Previous studies have shown" in output
        assert "Understanding light effects" in output
        assert "1. Measure growth rates" in output
        assert "2. Analyze chlorophyll content" in output
        assert "3. Compare root development" in output
        
        # Check methods
        assert "Controlled environment experiments" in output
        assert "* Seeds" in output
        assert "* Growth chambers" in output
        assert "* Light meters" in output
        assert "1. Prepare growth chambers" in output
        
        # Check discussion
        assert "Light intensity significantly affects" in output
        assert "The results confirm" in output
        assert "* Limited to laboratory conditions" in output
        
        # Check conclusion
        assert "Light intensity significantly affects" in output
        assert "* Extend to multiple species" in output
        
        # Check references
        assert "1. Smith et al. (2020). Plant Growth Studies." in output
        assert "2. Johnson (2019). Light Effects on Plants." in output
    
    def test_results_with_data(self):
        """Test results with various data formats."""
        results = [
            {
                "description": "Temperature measurement",
                "data": {"temperature": 25.5, "humidity": 60}
            },
            {
                "description": "Chemical analysis",
                "data": {"concentration": 0.15, "pH": 7.2}
            },
            {
                "description": "Simple measurement",
                "data": 42.0
            }
        ]
        context = {}
        
        output = draft(results, context)
        
        assert "Temperature measurement" in output
        assert "* temperature: 25.5" in output
        assert "* humidity: 60" in output
        assert "Chemical analysis" in output
        assert "* concentration: 0.15" in output
        assert "* pH: 7.2" in output
        assert "Simple measurement" in output
        assert "* Value: 42.0" in output
    
    def test_error_handling(self):
        """Test error handling for invalid input types."""
        # Test with wrong types
        with pytest.raises(ValueError, match="results must be a list"):
            draft("not a list", {})
        
        with pytest.raises(ValueError, match="context must be a dictionary"):
            draft([], "not a dict")
        
        with pytest.raises(ValueError, match="results must be a list"):
            draft(None, {})
        
        with pytest.raises(ValueError, match="context must be a dictionary"):
            draft([], None)
    
    def test_markdown_structure(self):
        """Test that output has proper Markdown structure."""
        results = [{"description": "Test result"}]
        context = {"objective": "Test objective"}
        
        output = draft(results, context)
        
        # Check for proper header structure
        lines = output.split('\n')
        assert any(line.startswith('# ') for line in lines)  # Title
        assert any(line.startswith('## ') for line in lines)  # Section headers
        
        # Check for bullet points
        assert '*' in output or '-' in output
    
    def test_no_references(self):
        """Test when no references are provided."""
        results = [{"description": "Test"}]
        context = {"objective": "Test"}
        
        output = draft(results, context)
        
        # Should not have references section
        assert "## References" not in output
    
    def test_empty_references(self):
        """Test when references list is empty."""
        results = [{"description": "Test"}]
        context = {"objective": "Test", "references": []}
        
        output = draft(results, context)
        
        # Should not have references section
        assert "## References" not in output
    
    def test_string_references(self):
        """Test when references is a string instead of list."""
        results = [{"description": "Test"}]
        context = {"objective": "Test", "references": "Single reference"}
        
        output = draft(results, context)
        
        assert "## References" in output
        assert "1. Single reference" in output
    
    def test_complex_results(self):
        """Test with complex result structures."""
        results = [
            {
                "id": "exp1",
                "description": "Complex experiment",
                "data": {"metric1": 100, "metric2": 200},
                "implication": "Significant finding",
                "significance": "High impact result"
            },
            {
                "id": "exp2", 
                "outcome": "Negative control",
                "finding": "Expected baseline",
                "metadata": {"timestamp": "2023-01-01"}
            }
        ]
        context = {"study_type": "Complex"}
        
        output = draft(results, context)
        
        assert "Complex experiment" in output
        assert "* metric1: 100" in output
        assert "* metric2: 200" in output
        assert "Negative control" in output
        assert "Expected baseline" in output
        assert "Significant finding" in output
        assert "High impact result" in output
    
    def test_title_generation(self):
        """Test various title generation scenarios."""
        # Test with study_type and subject
        context1 = {"study_type": "Randomized", "subject": "Clinical Trial"}
        output1 = draft([], context1)
        assert "Randomized on Clinical Trial" in output1
        
        # Test with objective containing capitalized words
        context2 = {"objective": "Study the Effects of Temperature on Growth"}
        output2 = draft([], context2)
        assert "Effects Temperature" in output2 or "Experimental Study" in output2
        
        # Test with minimal context
        context3 = {}
        output3 = draft([], context3)
        assert "Experimental Study" in output3
    
    def test_abstract_generation(self):
        """Test abstract generation with various inputs."""
        # Test with objective and methods
        context1 = {
            "objective": "test hypothesis",
            "methods": "standard procedures",
            "conclusion": "hypothesis confirmed"
        }
        results1 = [{"outcome": "positive result"}]
        output1 = draft(results1, context1)
        
        assert "**Background:** test hypothesis" in output1
        assert "**Methods:** standard procedures" in output1
        assert "**Results:** positive result" in output1
        assert "**Conclusion:** hypothesis confirmed" in output1
        
        # Test with multiple results
        results2 = [{"outcome": "result1"}, {"outcome": "result2"}]
        context2 = {"objective": "test"}
        output2 = draft(results2, context2)
        
        assert "2 experimental conditions were analyzed" in output2
    
    def test_methods_generation(self):
        """Test methods section generation."""
        context = {
            "methods": "Advanced techniques used",
            "materials": ["Material A", "Material B"],
            "procedures": ["Step 1", "Step 2", "Step 3"]
        }
        
        output = draft([], context)
        
        assert "Advanced techniques used" in output
        assert "* Material A" in output
        assert "* Material B" in output
        assert "1. Step 1" in output
        assert "2. Step 2" in output
        assert "3. Step 3" in output
    
    def test_discussion_generation(self):
        """Test discussion section generation."""
        results = [
            {"implication": "Important discovery"},
            {"significance": "High impact"}
        ]
        context = {
            "interpretation": "Results suggest pattern",
            "limitations": ["Limited sample size", "Short duration"]
        }
        
        output = draft(results, context)
        
        assert "**Key Findings:**" in output
        assert "* Finding 1: Important discovery" in output
        assert "* Finding 2: High impact" in output
        assert "**Interpretation:** Results suggest pattern" in output
        assert "**Limitations:**" in output
        assert "* Limited sample size" in output
        assert "* Short duration" in output
    
    def test_conclusion_generation(self):
        """Test conclusion section generation."""
        context = {
            "conclusion": "Study confirms hypothesis",
            "future_work": ["Extend study", "More samples"]
        }
        
        output = draft([], context)
        
        assert "Study confirms hypothesis" in output
        assert "**Future Directions:**" in output
        assert "* Extend study" in output
        assert "* More samples" in output


if __name__ == "__main__":
    pytest.main([__file__])