"""
Scientific Critic Agent for quality control and bias detection.
"""

import logging
from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ScientificCriticAgent(BaseAgent):
    """
    Scientific Critic Agent that evaluates research outputs for logical consistency,
    bias detection, and methodological rigor.
    """
    
    def __init__(self, agent_id: str = "scientific_critic",
                 model_config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id=agent_id,
            role="Scientific Critic",
            expertise=[
                "Research Methodology", "Critical Analysis", "Bias Detection",
                "Logical Reasoning", "Quality Assessment", "Scientific Rigor"
            ],
            model_config=model_config
        )
        self.critique_history = []
    
    def generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate critical analysis response."""
        response = f"""
        Critical analysis of: "{prompt}"
        
        Methodological Assessment:
        - Evaluate research design appropriateness
        - Assess sample size and selection bias
        - Review statistical analysis methods
        
        Logical Consistency:
        - Check for internal contradictions
        - Evaluate cause-effect relationships
        - Assess evidence-conclusion alignment
        
        Bias Detection:
        - Identify potential confirmation bias
        - Evaluate researcher objectivity
        - Check for systematic errors
        
        Quality Recommendations:
        - Suggest methodology improvements
        - Recommend additional controls
        - Propose validation approaches
        """
        return response
    
    def assess_task_relevance(self, task_description: str) -> float:
        """Scientific critic is relevant to all research evaluation tasks."""
        critique_keywords = [
            'evaluate', 'assess', 'critique', 'review', 'quality', 'bias',
            'methodology', 'validity', 'reliability', 'rigor', 'logical',
            'critical', 'analysis', 'error', 'limitation'
        ]
        
        task_lower = task_description.lower()
        matches = sum(1 for keyword in critique_keywords if keyword in task_lower)
        # Critic has high relevance for quality control tasks
        return min(1.0, matches * 0.2 + 0.3)  # Base relevance of 0.3
    
    def critique_research_output(self, output_content: str, 
                               output_type: str = "general",
                               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive critique of research output.
        
        Args:
            output_content: The research output to critique
            output_type: Type of output (manuscript, analysis, proposal, etc.)
            context: Additional context for the critique
            
        Returns:
            Comprehensive critique dictionary
        """
        logger.info(f"Conducting critique of {output_type} output")
        
        critique = {
            'output_type': output_type,
            'critique_id': f"critique_{len(self.critique_history) + 1}",
            'timestamp': self._get_timestamp(),
            'content_length': len(output_content),
            'context': context or {}
        }
        
        # Perform different types of analysis
        critique['logical_consistency'] = self._assess_logical_consistency(output_content)
        critique['methodological_rigor'] = self._assess_methodological_rigor(output_content)
        critique['bias_detection'] = self._detect_bias(output_content)
        critique['evidence_quality'] = self._assess_evidence_quality(output_content)
        critique['clarity_assessment'] = self._assess_clarity(output_content)
        
        # Calculate overall quality score
        critique['overall_score'] = self._calculate_overall_score(critique)
        
        # Generate recommendations
        critique['recommendations'] = self._generate_recommendations(critique)
        
        # Flag critical issues
        critique['critical_issues'] = self._identify_critical_issues(critique)
        
        # Store critique
        self.critique_history.append(critique)
        
        logger.info(f"Critique completed. Overall score: {critique['overall_score']}/100")
        return critique
    
    def _assess_logical_consistency(self, content: str) -> Dict[str, Any]:
        """Assess logical consistency of the content."""
        # Simplified implementation - real version would use sophisticated NLP
        logical_flags = []
        score = 85  # Default good score
        
        # Check for common logical issues
        if 'therefore' in content.lower() and 'because' not in content.lower():
            logical_flags.append("Conclusion without clear reasoning")
            score -= 10
        
        if content.count('however') > 3:
            logical_flags.append("Multiple contradictory statements")
            score -= 5
        
        return {
            'score': max(0, score),
            'flags': logical_flags,
            'assessment': 'Good logical flow' if score > 75 else 'Logical issues detected'
        }
    
    def _assess_methodological_rigor(self, content: str) -> Dict[str, Any]:
        """Assess methodological rigor."""
        rigor_keywords = [
            'method', 'procedure', 'protocol', 'control', 'sample',
            'statistical', 'significance', 'power', 'validation'
        ]
        
        content_lower = content.lower()
        keyword_count = sum(1 for keyword in rigor_keywords if keyword in content_lower)
        
        score = min(100, keyword_count * 10 + 50)
        
        flags = []
        if 'sample size' not in content_lower:
            flags.append("Sample size not specified")
            score -= 10
        
        if 'control' not in content_lower:
            flags.append("Control conditions unclear")
            score -= 10
        
        return {
            'score': max(0, score),
            'flags': flags,
            'keyword_count': keyword_count,
            'assessment': 'Rigorous methodology' if score > 75 else 'Methodology needs improvement'
        }
    
    def _detect_bias(self, content: str) -> Dict[str, Any]:
        """Detect potential bias in the content."""
        bias_indicators = []
        score = 90  # Start with good score
        
        # Check for absolute language
        absolute_words = ['always', 'never', 'all', 'none', 'completely', 'totally']
        absolute_count = sum(1 for word in absolute_words if word in content.lower())
        
        if absolute_count > 2:
            bias_indicators.append("Excessive absolute language")
            score -= 15
        
        # Check for confirmation bias indicators
        if 'confirms' in content.lower() and 'contradicts' not in content.lower():
            bias_indicators.append("Potential confirmation bias")
            score -= 10
        
        # Check for balanced perspective
        if 'limitation' not in content.lower():
            bias_indicators.append("No limitations discussed")
            score -= 10
        
        return {
            'score': max(0, score),
            'indicators': bias_indicators,
            'absolute_language_count': absolute_count,
            'assessment': 'Low bias detected' if score > 75 else 'Potential bias issues'
        }
    
    def _assess_evidence_quality(self, content: str) -> Dict[str, Any]:
        """Assess quality of evidence presented."""
        evidence_keywords = [
            'study', 'research', 'data', 'evidence', 'finding',
            'result', 'analysis', 'peer-reviewed', 'published'
        ]
        
        content_lower = content.lower()
        evidence_count = sum(1 for keyword in evidence_keywords if keyword in content_lower)
        
        score = min(100, evidence_count * 8 + 40)
        
        quality_flags = []
        if 'peer-reviewed' not in content_lower:
            quality_flags.append("Peer-review status unclear")
            score -= 5
        
        if content_lower.count('study') < 2:
            quality_flags.append("Limited research citations")
            score -= 10
        
        return {
            'score': max(0, score),
            'flags': quality_flags,
            'evidence_references': evidence_count,
            'assessment': 'Strong evidence base' if score > 75 else 'Evidence needs strengthening'
        }
    
    def _assess_clarity(self, content: str) -> Dict[str, Any]:
        """Assess clarity and readability."""
        # Simple readability metrics
        sentences = content.count('.') + content.count('!') + content.count('?')
        words = len(content.split())
        
        avg_sentence_length = words / max(1, sentences)
        
        clarity_flags = []
        score = 80
        
        if avg_sentence_length > 25:
            clarity_flags.append("Sentences too long")
            score -= 15
        
        if words < 100:
            clarity_flags.append("Content too brief")
            score -= 10
        
        # Check for jargon density
        technical_indicators = content.count('(') + content.count('i.e.') + content.count('e.g.')
        if technical_indicators / max(1, words) > 0.05:
            clarity_flags.append("High jargon density")
            score -= 10
        
        return {
            'score': max(0, score),
            'flags': clarity_flags,
            'avg_sentence_length': avg_sentence_length,
            'word_count': words,
            'assessment': 'Clear and readable' if score > 70 else 'Clarity needs improvement'
        }
    
    def _calculate_overall_score(self, critique: Dict[str, Any]) -> int:
        """Calculate overall quality score."""
        scores = [
            critique['logical_consistency']['score'],
            critique['methodological_rigor']['score'], 
            critique['bias_detection']['score'],
            critique['evidence_quality']['score'],
            critique['clarity_assessment']['score']
        ]
        
        # Weighted average
        weights = [0.25, 0.25, 0.2, 0.2, 0.1]
        weighted_score = sum(score * weight for score, weight in zip(scores, weights))
        
        return int(weighted_score)
    
    def _generate_recommendations(self, critique: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        if critique['logical_consistency']['score'] < 75:
            recommendations.append("Improve logical flow and argumentation structure")
        
        if critique['methodological_rigor']['score'] < 75:
            recommendations.append("Strengthen methodological description and controls")
        
        if critique['bias_detection']['score'] < 75:
            recommendations.append("Address potential bias and add balanced perspective")
        
        if critique['evidence_quality']['score'] < 75:
            recommendations.append("Strengthen evidence base with peer-reviewed sources")
        
        if critique['clarity_assessment']['score'] < 70:
            recommendations.append("Improve clarity and readability")
        
        if not recommendations:
            recommendations.append("Excellent quality - consider minor refinements only")
        
        return recommendations
    
    def _identify_critical_issues(self, critique: Dict[str, Any]) -> List[str]:
        """Identify critical issues that need immediate attention."""
        critical_issues = []
        
        if critique['overall_score'] < 60:
            critical_issues.append("Overall quality below acceptable threshold")
        
        if critique['logical_consistency']['score'] < 50:
            critical_issues.append("Serious logical inconsistencies detected")
        
        if critique['bias_detection']['score'] < 50:
            critical_issues.append("Significant bias concerns")
        
        return critical_issues
    
    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()
    
    def get_critique_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get critique history."""
        if limit:
            return self.critique_history[-limit:]
        return self.critique_history.copy()
    
    def get_critique_statistics(self) -> Dict[str, Any]:
        """Get statistics about critiques performed."""
        if not self.critique_history:
            return {'total_critiques': 0}
        
        scores = [c['overall_score'] for c in self.critique_history]
        
        return {
            'total_critiques': len(self.critique_history),
            'average_score': sum(scores) / len(scores),
            'highest_score': max(scores),
            'lowest_score': min(scores),
            'critiques_above_75': sum(1 for score in scores if score > 75),
            'critical_issues_flagged': sum(1 for c in self.critique_history if c['critical_issues'])
        }