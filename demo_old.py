#!/usr/bin/env python3
"""
Demonstration script for manuscript_drafter module.

Shows how to use the module to generate a scientific manuscript draft
from experimental results and contextual information.
"""

from manuscript_drafter import draft

def main():
    """Demonstrate the manuscript drafter functionality."""
    
    # Example experimental results
    results = [
        {
            "description": "Temperature effect on enzyme activity",
            "data": {
                "temperature_20C": 0.45,
                "temperature_30C": 0.78,
                "temperature_40C": 0.92,
                "temperature_50C": 0.67
            },
            "implication": "Optimal enzyme activity occurs at 40°C",
            "significance": "Temperature control is critical for enzymatic reactions"
        },
        {
            "description": "pH effect on enzyme stability",
            "data": {
                "pH_5": 0.23,
                "pH_6": 0.67,
                "pH_7": 0.89,
                "pH_8": 0.76,
                "pH_9": 0.34
            },
            "implication": "Enzyme shows maximum stability at pH 7",
            "significance": "pH optimization is essential for enzyme applications"
        },
        {
            "description": "Substrate concentration kinetics",
            "data": {
                "km": 0.15,
                "vmax": 2.34,
                "r_squared": 0.98
            },
            "implication": "Michaelis-Menten kinetics observed",
            "significance": "Classical enzyme kinetics model applies"
        }
    ]
    
    # Example contextual information
    context = {
        "study_type": "Enzymatic",
        "subject": "Enzyme Kinetics",
        "objective": "Characterize the temperature and pH dependence of enzyme activity",
        "background": "Enzymes are biological catalysts that accelerate chemical reactions. Understanding their activity under different conditions is crucial for industrial applications and biochemical research.",
        "significance": "This study provides essential data for optimizing enzyme-based processes in biotechnology and pharmaceutical industries.",
        "objectives": [
            "Determine optimal temperature for enzyme activity",
            "Identify pH range for maximum enzyme stability",
            "Establish kinetic parameters using Michaelis-Menten analysis",
            "Compare experimental results with theoretical predictions"
        ],
        "methods": "Enzyme activity was measured spectrophotometrically using a standard assay protocol. Temperature and pH conditions were systematically varied while maintaining constant substrate concentration.",
        "materials": [
            "Purified enzyme preparation",
            "Substrate solution (1 mM)",
            "Buffer solutions (pH 5-9)",
            "Spectrophotometer",
            "Temperature-controlled water bath"
        ],
        "procedures": [
            "Prepare enzyme solution in appropriate buffer",
            "Set temperature using water bath",
            "Add substrate and initiate reaction",
            "Monitor absorbance change over time",
            "Calculate initial reaction rates",
            "Repeat for different temperature and pH conditions"
        ],
        "conclusion": "The enzyme exhibits optimal activity at 40°C and pH 7, with classical Michaelis-Menten kinetics observed.",
        "interpretation": "The temperature and pH optima suggest this enzyme evolved for mesophilic conditions and neutral pH environments.",
        "limitations": [
            "Study limited to single enzyme preparation",
            "Substrate specificity not fully characterized",
            "Long-term stability not assessed",
            "Industrial conditions not simulated"
        ],
        "future_work": [
            "Extend to multiple enzyme isoforms",
            "Investigate substrate specificity",
            "Assess long-term stability under various conditions",
            "Test performance in industrial process conditions"
        ],
        "references": [
            {
                "authors": "Smith et al.",
                "year": "2020",
                "title": "Enzyme Kinetics: Principles and Applications"
            },
            {
                "authors": "Johnson and Brown",
                "year": "2019",
                "title": "Temperature Effects on Enzyme Activity"
            },
            {
                "authors": "Wilson",
                "year": "2021",
                "title": "pH Dependence in Enzymatic Reactions"
            }
        ]
    }
    
    # Generate the manuscript draft
    print("Generating scientific manuscript draft...\n")
    manuscript = draft(results, context)
    
    # Print the generated manuscript
    print(manuscript)
    
    # Print summary statistics
    print(f"\n{'='*50}")
    print("SUMMARY STATISTICS:")
    print(f"{'='*50}")
    print(f"Total manuscript length: {len(manuscript)} characters")
    print(f"Number of experimental results: {len(results)}")
    print(f"Number of context fields: {len(context)}")
    print(f"Number of references: {len(context.get('references', []))}")
    
    # Check for key sections
    sections = ["#", "## Abstract", "## Introduction", "## Methods", "## Results", "## Discussion", "## Conclusion", "## References"]
    found_sections = [section for section in sections if section in manuscript]
    print(f"Manuscript sections found: {len(found_sections)}/{len(sections)}")
    
    print(f"\n{'='*50}")
    print("DEMONSTRATION COMPLETE")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()