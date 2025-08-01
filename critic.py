"""
Critic module for reviewing and critiquing research outputs.

This module provides rule-based critique functionality for research outputs,
designed to be easily replaceable with LLM-based approaches in the future.
"""

import re
from typing import Dict, List, Union


class Critic:
    """
    A critic class that analyzes research outputs and provides structured feedback.
    
    The critic uses rule-based heuristics to evaluate research content and provide
    constructive feedback including strengths, weaknesses, and improvement suggestions.
    """
    
    def __init__(self):
        """Initialize the Critic with default heuristic thresholds."""
        self.min_word_count = 50
        self.max_word_count = 10000
        self.min_paragraph_count = 2
        self.max_repetition_ratio = 0.3
        
    def review(self, output: str) -> Dict[str, Union[str, List[str]]]:
        """
        Review a research output and provide structured critique.
        
        Args:
            output (str): The research output text to review
            
        Returns:
            dict: A dictionary containing:
                - strengths: List of identified strengths
                - weaknesses: List of identified weaknesses 
                - suggestions: List of improvement suggestions
                - overall_score: Numerical score (0-100)
        """
        if not isinstance(output, str):
            raise ValueError("Output must be a string")
            
        if not output.strip():
            return {
                "strengths": [],
                "weaknesses": ["Empty or whitespace-only content"],
                "suggestions": ["Provide actual content to review"],
                "overall_score": 0
            }
            
        # Analyze the output using various heuristics
        analysis = self._analyze_content(output)
        
        # Generate critique based on analysis
        strengths = self._identify_strengths(analysis)
        weaknesses = self._identify_weaknesses(analysis)
        suggestions = self._generate_suggestions(analysis, weaknesses)
        score = self._calculate_score(analysis)
        
        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "suggestions": suggestions,
            "overall_score": score
        }
    
    def _analyze_content(self, text: str) -> Dict[str, Union[int, float, bool]]:
        """Analyze content using various heuristics."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Basic metrics
        word_count = len(words)
        sentence_count = len([s for s in sentences if s.strip()])
        paragraph_count = len(paragraphs)
        
        # Content analysis
        has_citations = bool(re.search(r'\[[0-9,\s]+\]|\([^)]*[0-9]{4}[^)]*\)|et al\.', text))
        has_methodology = bool(re.search(r'\b(method|approach|technique|procedure|analysis)\b', text, re.IGNORECASE))
        has_results = bool(re.search(r'\b(result|finding|outcome|conclusion|evidence)\b', text, re.IGNORECASE))
        has_structured_sections = bool(re.search(r'\b(introduction|methodology|results?|discussion|conclusion)\b', text, re.IGNORECASE))
        
        # Quality indicators
        avg_sentence_length = word_count / max(sentence_count, 1)
        repetition_ratio = self._calculate_repetition_ratio(words)
        academic_terms = len(re.findall(r'\b(research|study|analysis|hypothesis|evidence|data|significant|correlation)\b', text, re.IGNORECASE))
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'avg_sentence_length': avg_sentence_length,
            'repetition_ratio': repetition_ratio,
            'has_citations': has_citations,
            'has_methodology': has_methodology,
            'has_results': has_results,
            'has_structured_sections': has_structured_sections,
            'academic_terms_count': academic_terms
        }
    
    def _calculate_repetition_ratio(self, words: List[str]) -> float:
        """Calculate the ratio of repeated words to total words."""
        if not words:
            return 0.0
            
        word_freq = {}
        for word in words:
            clean_word = word.lower().strip('.,!?;:')
            if len(clean_word) > 2:  # Only count words longer than 2 characters
                word_freq[clean_word] = word_freq.get(clean_word, 0) + 1
        
        repeated_words = sum(count - 1 for count in word_freq.values() if count > 1)
        return repeated_words / len(words) if words else 0.0
    
    def _identify_strengths(self, analysis: Dict) -> List[str]:
        """Identify strengths based on content analysis."""
        strengths = []
        
        if analysis['word_count'] >= self.min_word_count:
            strengths.append("Adequate content length for substantive analysis")
            
        if analysis['has_citations']:
            strengths.append("Includes citations or references to support claims")
            
        if analysis['has_methodology']:
            strengths.append("Discusses methodology or approach")
            
        if analysis['has_results']:
            strengths.append("Presents results or findings")
            
        if analysis['has_structured_sections']:
            strengths.append("Uses structured academic sections")
            
        if analysis['paragraph_count'] >= self.min_paragraph_count:
            strengths.append("Well-organized with multiple paragraphs")
            
        if 15 <= analysis['avg_sentence_length'] <= 25:
            strengths.append("Good sentence length variety for readability")
            
        if analysis['repetition_ratio'] < 0.15:
            strengths.append("Minimal unnecessary repetition")
            
        if analysis['academic_terms_count'] >= 3:
            strengths.append("Uses appropriate academic terminology")
            
        return strengths
    
    def _identify_weaknesses(self, analysis: Dict) -> List[str]:
        """Identify weaknesses based on content analysis."""
        weaknesses = []
        
        if analysis['word_count'] < self.min_word_count:
            weaknesses.append("Content too short for comprehensive analysis")
            
        if analysis['word_count'] > self.max_word_count:
            weaknesses.append("Content may be too lengthy and verbose")
            
        if not analysis['has_citations']:
            weaknesses.append("Lacks citations or references to support claims")
            
        if not analysis['has_methodology']:
            weaknesses.append("Missing clear methodology or approach description")
            
        if not analysis['has_results']:
            weaknesses.append("No clear results or findings presented")
            
        if analysis['paragraph_count'] < self.min_paragraph_count:
            weaknesses.append("Poor organization - needs more paragraph structure")
            
        if analysis['avg_sentence_length'] < 10:
            weaknesses.append("Sentences too short - may lack complexity")
            
        if analysis['avg_sentence_length'] > 30:
            weaknesses.append("Sentences too long - may hurt readability")
            
        if analysis['repetition_ratio'] > self.max_repetition_ratio:
            weaknesses.append("Excessive repetition of words and phrases")
            
        if analysis['academic_terms_count'] < 2:
            weaknesses.append("Limited use of academic terminology")
            
        return weaknesses
    
    def _generate_suggestions(self, analysis: Dict, weaknesses: List[str]) -> List[str]:
        """Generate improvement suggestions based on analysis and weaknesses."""
        suggestions = []
        
        if analysis['word_count'] < self.min_word_count:
            suggestions.append("Expand content with more detailed analysis and examples")
            
        if analysis['word_count'] > self.max_word_count:
            suggestions.append("Consider condensing content and removing redundant information")
            
        if not analysis['has_citations']:
            suggestions.append("Add citations and references to support key claims")
            
        if not analysis['has_methodology']:
            suggestions.append("Include a clear description of research methodology")
            
        if not analysis['has_results']:
            suggestions.append("Present clear results, findings, or outcomes")
            
        if analysis['paragraph_count'] < self.min_paragraph_count:
            suggestions.append("Break content into logical paragraphs for better organization")
            
        if analysis['repetition_ratio'] > self.max_repetition_ratio:
            suggestions.append("Reduce repetition by using synonyms and varied phrasing")
            
        if not analysis['has_structured_sections']:
            suggestions.append("Consider using standard academic structure (Introduction, Methods, Results, Discussion)")
            
        if analysis['academic_terms_count'] < 2:
            suggestions.append("Incorporate more academic terminology appropriate to the field")
            
        # Always provide at least one constructive suggestion
        if not suggestions:
            suggestions.append("Consider adding more specific examples or case studies to strengthen arguments")
            
        return suggestions
    
    def _calculate_score(self, analysis: Dict) -> int:
        """Calculate an overall quality score (0-100) based on analysis."""
        score = 50  # Base score
        
        # Word count scoring
        if self.min_word_count <= analysis['word_count'] <= self.max_word_count:
            score += 10
        elif analysis['word_count'] < self.min_word_count // 2:
            score -= 20
            
        # Structure scoring
        if analysis['paragraph_count'] >= self.min_paragraph_count:
            score += 10
        if analysis['has_structured_sections']:
            score += 15
            
        # Content quality scoring
        if analysis['has_citations']:
            score += 15
        if analysis['has_methodology']:
            score += 10
        if analysis['has_results']:
            score += 10
            
        # Readability scoring
        if 15 <= analysis['avg_sentence_length'] <= 25:
            score += 5
        if analysis['repetition_ratio'] < 0.15:
            score += 10
        elif analysis['repetition_ratio'] > self.max_repetition_ratio:
            score -= 15
            
        # Academic quality
        if analysis['academic_terms_count'] >= 3:
            score += 10
        elif analysis['academic_terms_count'] < 1:
            score -= 10
            
        return max(0, min(100, score))