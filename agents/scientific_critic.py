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
                 model_config: Optional[Dict[str, Any]] = None, cost_manager=None):
        super().__init__(
            agent_id=agent_id,
            role="Scientific Critic",
            expertise=[
                "Research Methodology", "Critical Analysis", "Bias Detection",
                "Logical Reasoning", "Quality Assessment", "Scientific Rigor"
            ],
            model_config=model_config,
            cost_manager=cost_manager
        )
        self.critique_history = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'critique_history_count': len(self.critique_history)
        })
        return base_dict
    
    def generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate critical analysis response."""
        specialized_prompt = f"""
        You are a scientific critic with expertise in research methodology, logical reasoning, and bias detection.
        Provide a rigorous critical analysis of the following research content:
        
        {prompt}
        
        Evaluate and provide feedback on:
        - Methodological rigor and appropriateness
        - Logical consistency and internal coherence
        - Potential biases and limitations
        - Evidence quality and strength
        - Statistical validity and interpretation
        - Research design strengths and weaknesses
        - Recommendations for improvement
        
        Be constructive but thorough in identifying potential issues and suggesting improvements.
        """
        
        # Add agent context for cost tracking
        context_with_agent = {
            **context,
            'agent_id': self.agent_id,
            'task_type': context.get('task_type', 'critique')
        }
        
        return self.llm_client.generate_response(
            specialized_prompt,
            context_with_agent,
            agent_role=self.role,
            cost_manager=self.cost_manager
        )
    
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
        """Assess logical consistency of the content using LLM analysis."""
        
        consistency_prompt = f"""
        Analyze the logical consistency of this research content. Look for:
        - Logical flow and reasoning
        - Contradictions or inconsistencies
        - Unsupported claims or conclusions
        - Clear argumentation structure
        
        Content to analyze:
        {content}
        
        Provide a score (0-100) and identify specific logical issues if any.
        
        Format response as:
        SCORE: [0-100]
        ISSUES: [issue1 | issue2 | issue3] (or "None identified")
        ASSESSMENT: [brief overall assessment]
        """
        
        llm_response = self.llm_client.generate_response(
            consistency_prompt, 
            {'content': content}, 
            agent_role="Logic Assessment Expert"
        )
        
        return self._parse_assessment_response(llm_response, default_score=85)
    
    def _assess_methodological_rigor(self, content: str) -> Dict[str, Any]:
        """Assess methodological rigor using LLM analysis."""
        
        rigor_prompt = f"""
        Evaluate the methodological rigor of this research content. Assess:
        - Research design and methodology
        - Sample size and selection criteria
        - Control conditions and variables
        - Statistical analysis approaches
        - Validation methods
        - Reproducibility considerations
        
        Content to evaluate:
        {content}
        
        Provide a score (0-100) and identify methodological strengths/weaknesses.
        
        Format response as:
        SCORE: [0-100]
        ISSUES: [issue1 | issue2 | issue3] (or "None identified")
        ASSESSMENT: [brief overall assessment]
        """
        
        llm_response = self.llm_client.generate_response(
            rigor_prompt, 
            {'content': content}, 
            agent_role="Methodology Assessment Expert"
        )
        
        return self._parse_assessment_response(llm_response, default_score=75)
    
    def _parse_assessment_response(self, response: str, default_score: int = 75) -> Dict[str, Any]:
        """Parse assessment response from LLM."""
        import re
        
        result = {
            'score': default_score,
            'flags': [],
            'assessment': 'Assessment completed'
        }
        
        try:
            # Extract score
            score_match = re.search(r'SCORE:\s*(\d+)', response)
            if score_match:
                result['score'] = max(0, min(100, int(score_match.group(1))))
            
            # Extract issues/flags
            issues_match = re.search(r'ISSUES:\s*([^\n]+)', response)
            if issues_match:
                issues_str = issues_match.group(1).strip()
                if issues_str.lower() != "none identified":
                    result['flags'] = [i.strip() for i in issues_str.split('|')]
            
            # Extract assessment
            assessment_match = re.search(r'ASSESSMENT:\s*([^\n]+)', response)
            if assessment_match:
                result['assessment'] = assessment_match.group(1).strip()
                
        except Exception as e:
            logger.warning(f"Failed to parse assessment response: {e}")
        
        return result
    
    def _detect_bias(self, content: str) -> Dict[str, Any]:
        """Detect potential bias in the content using LLM analysis."""
        
        bias_prompt = f"""
        Analyze this research content for potential bias. Look for:
        - Confirmation bias (only presenting supporting evidence)
        - Selection bias in data or sources
        - Language bias (absolute statements, emotional language)
        - Cultural or demographic bias
        - Funding or conflict of interest bias
        - Balanced perspective and limitations discussion
        
        Content to analyze:
        {content}
        
        Provide a score (0-100, where 100 is unbiased) and identify specific bias concerns.
        
        Format response as:
        SCORE: [0-100]
        ISSUES: [bias1 | bias2 | bias3] (or "None identified")
        ASSESSMENT: [brief overall assessment]
        """
        
        llm_response = self.llm_client.generate_response(
            bias_prompt, 
            {'content': content}, 
            agent_role="Bias Detection Expert"
        )
        
        return self._parse_assessment_response(llm_response, default_score=85)
    
    def _assess_evidence_quality(self, content: str) -> Dict[str, Any]:
        """Assess quality of evidence presented using LLM analysis."""
        
        evidence_prompt = f"""
        Evaluate the quality and strength of evidence in this research content. Assess:
        - Citation of peer-reviewed sources
        - Quality and recency of references
        - Strength of empirical evidence
        - Appropriate use of primary vs secondary sources
        - Statistical evidence and data quality
        - Reproducibility of findings
        
        Content to evaluate:
        {content}
        
        Provide a score (0-100) and identify evidence strengths/weaknesses.
        
        Format response as:
        SCORE: [0-100]
        ISSUES: [issue1 | issue2 | issue3] (or "None identified")
        ASSESSMENT: [brief overall assessment]
        """
        
        llm_response = self.llm_client.generate_response(
            evidence_prompt, 
            {'content': content}, 
            agent_role="Evidence Quality Expert"
        )
        
        return self._parse_assessment_response(llm_response, default_score=75)
    
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