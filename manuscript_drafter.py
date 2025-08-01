"""
Scientific Manuscript Drafter

A module for generating structured Markdown drafts from experimental results
and contextual information.
"""

from typing import List, Dict, Any
import re


def draft(results: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
    """
    Generate a structured Markdown draft from experimental results and context.
    
    Args:
        results: List of dictionaries containing experimental results
        context: Dictionary containing contextual information for the manuscript
        
    Returns:
        str: Formatted Markdown string containing the manuscript draft
    """
    if not isinstance(results, list) or not isinstance(context, dict):
        raise ValueError("results must be a list and context must be a dictionary")
    
    # Initialize the manuscript content
    manuscript_parts = []
    
    # Generate title
    title = _generate_title(context)
    manuscript_parts.append(f"# {title}\n")
    
    # Generate abstract
    abstract = _generate_abstract(results, context)
    manuscript_parts.append(f"## Abstract\n\n{abstract}\n")
    
    # Generate introduction
    introduction = _generate_introduction(context)
    manuscript_parts.append(f"## Introduction\n\n{introduction}\n")
    
    # Generate methods
    methods = _generate_methods(context)
    manuscript_parts.append(f"## Methods\n\n{methods}\n")
    
    # Generate results
    results_section = _generate_results(results)
    manuscript_parts.append(f"## Results\n\n{results_section}\n")
    
    # Generate discussion
    discussion = _generate_discussion(results, context)
    manuscript_parts.append(f"## Discussion\n\n{discussion}\n")
    
    # Generate conclusion
    conclusion = _generate_conclusion(results, context)
    manuscript_parts.append(f"## Conclusion\n\n{conclusion}\n")
    
    # Generate references
    references = _generate_references(context)
    if references:
        manuscript_parts.append(f"## References\n\n{references}\n")
    
    return "\n".join(manuscript_parts)


def _generate_title(context: Dict[str, Any]) -> str:
    """Generate a title based on context information."""
    if not context:
        return "Experimental Study"
    
    # Try to extract meaningful title components
    title_components = []
    
    if "study_type" in context:
        title_components.append(context["study_type"])
    
    if "subject" in context:
        title_components.append(f"on {context['subject']}")
    
    if "objective" in context:
        # Extract key words from objective
        objective = context["objective"]
        key_words = re.findall(r'\b[A-Z][a-z]+\b', objective)
        if key_words:
            title_components.extend(key_words[:2])  # Use first 2 capitalized words
    
    if not title_components:
        title_components = ["Experimental", "Study"]
    
    return " ".join(title_components)


def _generate_abstract(results: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
    """Generate an abstract summarizing the study."""
    if not results and not context:
        return "This study presents experimental findings and their implications."
    
    abstract_parts = []
    
    # Background
    if "objective" in context:
        abstract_parts.append(f"**Background:** {context['objective']}")
    
    # Methods summary
    if "methods" in context:
        abstract_parts.append(f"**Methods:** {context['methods']}")
    
    # Results summary
    if results:
        result_summary = f"**Results:** {len(results)} experimental conditions were analyzed."
        if len(results) == 1:
            result = results[0]
            if "outcome" in result:
                result_summary = f"**Results:** {result['outcome']}"
        abstract_parts.append(result_summary)
    
    # Conclusion
    if "conclusion" in context:
        abstract_parts.append(f"**Conclusion:** {context['conclusion']}")
    else:
        abstract_parts.append("**Conclusion:** The findings provide insights into the experimental conditions studied.")
    
    return " ".join(abstract_parts)


def _generate_introduction(context: Dict[str, Any]) -> str:
    """Generate an introduction section."""
    if not context:
        return "This study investigates experimental conditions and their outcomes."
    
    intro_parts = []
    
    # Background
    if "background" in context:
        intro_parts.append(context["background"])
    elif "objective" in context:
        intro_parts.append(f"The objective of this study is to {context['objective']}.")
    else:
        intro_parts.append("This study examines experimental conditions and their effects.")
    
    # Significance
    if "significance" in context:
        intro_parts.append(f"This research is significant because {context['significance']}.")
    
    # Objectives
    if "objectives" in context:
        objectives = context["objectives"]
        if isinstance(objectives, list):
            intro_parts.append("The specific objectives of this study are:")
            for i, obj in enumerate(objectives, 1):
                intro_parts.append(f"{i}. {obj}")
        else:
            intro_parts.append(f"The objectives include: {objectives}")
    
    return "\n\n".join(intro_parts)


def _generate_methods(context: Dict[str, Any]) -> str:
    """Generate a methods section."""
    if not context:
        return "Standard experimental procedures were followed."
    
    methods_parts = []
    
    if "methods" in context:
        methods_parts.append(context["methods"])
    
    if "materials" in context:
        materials = context["materials"]
        if isinstance(materials, list):
            methods_parts.append("**Materials:**")
            for material in materials:
                methods_parts.append(f"* {material}")
        else:
            methods_parts.append(f"**Materials:** {materials}")
    
    if "procedures" in context:
        procedures = context["procedures"]
        if isinstance(procedures, list):
            methods_parts.append("**Procedures:**")
            for i, proc in enumerate(procedures, 1):
                methods_parts.append(f"{i}. {proc}")
        else:
            methods_parts.append(f"**Procedures:** {procedures}")
    
    if not methods_parts:
        methods_parts.append("Standard experimental procedures were followed with appropriate controls.")
    
    return "\n\n".join(methods_parts)


def _generate_results(results: List[Dict[str, Any]]) -> str:
    """Generate a results section from experimental results."""
    if not results:
        return "No experimental results were obtained."
    
    results_parts = []
    
    for i, result in enumerate(results, 1):
        result_text = f"**Result {i}:** "
        
        if "description" in result:
            result_text += result["description"]
        elif "outcome" in result:
            result_text += result["outcome"]
        elif "finding" in result:
            result_text += result["finding"]
        else:
            # Try to construct from available fields
            available_fields = [k for k in result.keys() if k not in ['id', 'timestamp', 'metadata']]
            if available_fields:
                field_values = [f"{k}: {result[k]}" for k in available_fields[:3]]  # Limit to 3 fields
                result_text += "; ".join(field_values)
            else:
                result_text += "Experimental condition analyzed."
        
        results_parts.append(result_text)
        
        # Add quantitative data if available
        if "data" in result:
            data = result["data"]
            if isinstance(data, dict):
                data_items = [f"* {k}: {v}" for k, v in data.items()]
                results_parts.extend(data_items)
            elif isinstance(data, (int, float)):
                results_parts.append(f"* Value: {data}")
    
    return "\n\n".join(results_parts)


def _generate_discussion(results: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
    """Generate a discussion section."""
    if not results and not context:
        return "The experimental findings warrant further investigation."
    
    discussion_parts = []
    
    # Summary of key findings
    if results:
        discussion_parts.append("**Key Findings:**")
        for i, result in enumerate(results, 1):
            if "implication" in result:
                discussion_parts.append(f"* Finding {i}: {result['implication']}")
            elif "significance" in result:
                discussion_parts.append(f"* Finding {i}: {result['significance']}")
            else:
                discussion_parts.append(f"* Finding {i}: Provides important experimental insights")
    
    # Interpretation
    if "interpretation" in context:
        discussion_parts.append(f"**Interpretation:** {context['interpretation']}")
    else:
        discussion_parts.append("**Interpretation:** The results suggest important patterns in the experimental conditions.")
    
    # Limitations
    if "limitations" in context:
        limitations = context["limitations"]
        if isinstance(limitations, list):
            discussion_parts.append("**Limitations:**")
            for limitation in limitations:
                discussion_parts.append(f"* {limitation}")
        else:
            discussion_parts.append(f"**Limitations:** {limitations}")
    else:
        discussion_parts.append("**Limitations:** This study has inherent limitations due to experimental constraints.")
    
    return "\n\n".join(discussion_parts)


def _generate_conclusion(results: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
    """Generate a conclusion section."""
    if not results and not context:
        return "This study provides a foundation for future research in this area."
    
    conclusion_parts = []
    
    # Main conclusions
    if "conclusion" in context:
        conclusion_parts.append(context["conclusion"])
    else:
        conclusion_parts.append("The experimental findings provide valuable insights into the studied phenomena.")
    
    # Future directions
    if "future_work" in context:
        future_work = context["future_work"]
        if isinstance(future_work, list):
            conclusion_parts.append("**Future Directions:**")
            for direction in future_work:
                conclusion_parts.append(f"* {direction}")
        else:
            conclusion_parts.append(f"**Future Directions:** {future_work}")
    else:
        conclusion_parts.append("**Future Directions:** Further research is needed to expand upon these findings.")
    
    return "\n\n".join(conclusion_parts)


def _generate_references(context: Dict[str, Any]) -> str:
    """Generate a references section."""
    if "references" not in context:
        return ""
    
    references = context["references"]
    if not references:
        return ""
    
    ref_parts = []
    
    if isinstance(references, list):
        for i, ref in enumerate(references, 1):
            if isinstance(ref, dict):
                # Handle structured reference
                if "authors" in ref and "title" in ref and "year" in ref:
                    ref_parts.append(f"{i}. {ref['authors']} ({ref['year']}). {ref['title']}.")
                elif "citation" in ref:
                    ref_parts.append(f"{i}. {ref['citation']}")
                else:
                    ref_parts.append(f"{i}. {str(ref)}")
            else:
                ref_parts.append(f"{i}. {ref}")
    else:
        ref_parts.append(f"1. {references}")
    
    return "\n".join(ref_parts)