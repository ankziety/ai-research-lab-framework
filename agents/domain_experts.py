"""
Domain Expert Agents for specialized research areas.
"""

import logging
from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class MedicalExpert(BaseAgent):
    """
    Expert agent specializing in medical and health sciences.
    Can be configured for specific medical domains.
    """
    
    def __init__(self, agent_id: str = "medical_expert", 
                 domain: str = "general_medicine",
                 model_config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id=agent_id,
            role=f"Medical Expert - {domain.replace('_', ' ').title()}",
            expertise=[
                "Medical Research", "Clinical Studies", "Health Assessment", 
                "Disease Mechanisms", "Treatment Protocols", "Patient Care",
                "Medical Literature", "Evidence-Based Medicine"
            ],
            model_config=model_config
        )
        self.domain = domain
    
    def generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate medical domain-focused response."""
        specialized_prompt = f"""
        You are a medical expert specializing in {self.domain.replace('_', ' ')}.
        Provide a detailed clinical and research perspective on the following:
        
        {prompt}
        
        Focus on:
        - Clinical assessment considerations
        - Diagnostic approaches
        - Research implications
        - Health-related quality of life factors
        - Evidence-based treatment options
        
        Base your response on current medical knowledge and best practices.
        """
        
        return self.llm_client.generate_response(
            specialized_prompt, 
            context, 
            agent_role=self.role
        )
    
    def assess_task_relevance(self, task_description: str) -> float:
        """Assess relevance to medical domain."""
        medical_keywords = [
            'health', 'medical', 'clinical', 'disease', 'treatment', 'patient',
            'diagnosis', 'therapy', 'medicine', 'healthcare', 'wellness',
            'syndrome', 'disorder', 'pathology', 'pharmacology', 'surgery'
        ]
        
        # Add domain-specific keywords
        domain_keywords = {
            'cardiology': ['heart', 'cardiac', 'cardiovascular', 'blood pressure'],
            'neurology': ['brain', 'neural', 'neurological', 'cognitive'],
            'psychiatry': ['mental', 'psychological', 'mood', 'anxiety', 'depression'],
            'ophthalmology': ['eye', 'vision', 'visual', 'retina', 'glaucoma', 'binocular'],
            'endocrinology': ['hormone', 'diabetes', 'thyroid', 'metabolism'],
            'oncology': ['cancer', 'tumor', 'malignant', 'chemotherapy']
        }
        
        all_keywords = medical_keywords + domain_keywords.get(self.domain, [])
        
        task_lower = task_description.lower()
        matches = sum(1 for keyword in all_keywords if keyword in task_lower)
        return min(1.0, matches * 0.15)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        base_dict = super().to_dict()
        base_dict['domain'] = self.domain
        return base_dict


class PsychologyExpert(BaseAgent):
    """
    Expert agent specializing in psychology and mental health.
    """
    
    def __init__(self, agent_id: str = "psychology_expert",
                 model_config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id=agent_id,
            role="Psychology Expert", 
            expertise=[
                "Clinical Psychology", "Mental Health", "Anxiety Disorders",
                "Depression", "Cognitive Psychology", "Behavioral Assessment"
            ],
            model_config=model_config
        )
    
    def generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate psychology-focused response."""
        specialized_prompt = f"""
        You are an expert clinical psychologist with extensive knowledge in mental health and behavioral science.
        Provide a comprehensive psychological analysis of the following:
        
        {prompt}
        
        Focus on:
        - Mental health considerations and assessment
        - Behavioral patterns and cognitive factors
        - Psychosocial impact and adaptation
        - Evidence-based intervention approaches
        - Psychological research implications
        
        Base your response on current psychological research and clinical best practices.
        """
        
        return self.llm_client.generate_response(
            specialized_prompt,
            context,
            agent_role=self.role
        )
    
    def assess_task_relevance(self, task_description: str) -> float:
        """Assess relevance to psychology domain."""
        psychology_keywords = [
            'psychology', 'mental health', 'anxiety', 'depression', 'stress',
            'cognitive', 'behavioral', 'therapy', 'psychological', 'emotion',
            'mood', 'coping', 'adaptation', 'wellbeing', 'psychiatric'
        ]
        
        task_lower = task_description.lower()
        matches = sum(1 for keyword in psychology_keywords if keyword in task_lower)
        return min(1.0, matches * 0.15)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return super().to_dict()


class NeuroscienceExpert(BaseAgent):
    """
    Expert agent specializing in neuroscience and neurological disorders.
    """
    
    def __init__(self, agent_id: str = "neuroscience_expert",
                 model_config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id=agent_id,
            role="Neuroscience Expert",
            expertise=[
                "Neuroscience", "Neurological Disorders", "Brain Function",
                "Neural Networks", "Neuroplasticity", "Cognitive Neuroscience"
            ],
            model_config=model_config
        )
    
    def generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate neuroscience-focused response."""
        specialized_prompt = f"""
        You are a neuroscience expert with deep knowledge in brain function, neurological disorders, and cognitive neuroscience.
        Provide a detailed neurological analysis of the following:
        
        {prompt}
        
        Focus on:
        - Neural mechanisms and brain pathways involved
        - Neurological assessment considerations
        - Cognitive and motor function implications
        - Current neuroscience research findings
        - Potential neuroplasticity factors
        
        Base your response on current neuroscientific knowledge and research.
        """
        
        return self.llm_client.generate_response(
            specialized_prompt,
            context,
            agent_role=self.role
        )
    
    def assess_task_relevance(self, task_description: str) -> float:
        """Assess relevance to neuroscience domain."""
        neuroscience_keywords = [
            'brain', 'neural', 'neuroscience', 'neurological', 'cortex',
            'neuron', 'synapse', 'cognition', 'neuroplasticity', 'EEG',
            'fMRI', 'neurotransmitter', 'cerebral', 'spinal', 'nervous system'
        ]
        
        task_lower = task_description.lower()
        matches = sum(1 for keyword in neuroscience_keywords if keyword in task_lower)
        return min(1.0, matches * 0.15)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return super().to_dict()


class DataScienceExpert(BaseAgent):
    """
    Expert agent specializing in data science and machine learning.
    """
    
    def __init__(self, agent_id: str = "data_science_expert",
                 model_config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id=agent_id,
            role="Data Science Expert",
            expertise=[
                "Data Science", "Machine Learning", "Statistical Analysis",
                "Data Mining", "Predictive Modeling", "Biostatistics"
            ],
            model_config=model_config
        )
    
    def generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate data science-focused response."""
        specialized_prompt = f"""
        You are a data science expert with extensive knowledge in statistical analysis, machine learning, and research methodology.
        Provide a comprehensive data science analysis of the following:
        
        {prompt}
        
        Focus on:
        - Statistical methods and analysis approaches
        - Machine learning applications and algorithms
        - Data visualization and interpretation
        - Research design and methodology considerations
        - Predictive modeling opportunities
        - Data quality and validation strategies
        
        Base your response on current data science best practices and methodologies.
        """
        
        return self.llm_client.generate_response(
            specialized_prompt,
            context,
            agent_role=self.role
        )
    
    def assess_task_relevance(self, task_description: str) -> float:
        """Assess relevance to data science domain."""
        data_science_keywords = [
            'data', 'analysis', 'machine learning', 'statistics', 'model',
            'algorithm', 'prediction', 'classification', 'regression',
            'clustering', 'mining', 'visualization', 'database', 'analytics'
        ]
        
        task_lower = task_description.lower()
        matches = sum(1 for keyword in data_science_keywords if keyword in task_lower)
        return min(1.0, matches * 0.15)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return super().to_dict()


class LiteratureResearcher(BaseAgent):
    """
    Expert agent specializing in literature research and review.
    """
    
    def __init__(self, agent_id: str = "literature_researcher",
                 model_config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id=agent_id,
            role="Literature Researcher",
            expertise=[
                "Literature Review", "Research Methodology", "Citation Analysis",
                "Systematic Reviews", "Meta-Analysis", "Academic Writing"
            ],
            model_config=model_config
        )
    
    def generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate literature research-focused response."""
        specialized_prompt = f"""
        You are a literature research expert with extensive knowledge in systematic reviews, research methodology, and academic writing.
        Provide a comprehensive literature research analysis of the following:
        
        {prompt}
        
        Focus on:
        - Literature search strategies and databases
        - Critical evaluation of research quality
        - Systematic review and meta-analysis approaches
        - Evidence synthesis and gap identification
        - Research methodology assessment
        - Citation analysis and academic writing standards
        
        Base your response on current standards for literature research and systematic reviews.
        """
        
        return self.llm_client.generate_response(
            specialized_prompt,
            context,
            agent_role=self.role
        )
    
    def assess_task_relevance(self, task_description: str) -> float:
        """Assess relevance to literature research domain."""
        literature_keywords = [
            'literature', 'review', 'papers', 'research', 'publications',
            'studies', 'systematic', 'meta-analysis', 'evidence', 'citation',
            'abstract', 'methodology', 'findings', 'knowledge', 'scholarly'
        ]
        
        task_lower = task_description.lower()
        matches = sum(1 for keyword in literature_keywords if keyword in task_lower)
        return min(1.0, matches * 0.15)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return super().to_dict()