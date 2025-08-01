#!/usr/bin/env python3
"""
AI-Powered Research Framework

A comprehensive multi-agent framework for AI-powered research workflows that coordinates
teams of AI expert agents to collaborate on research problems across any domain.

This framework includes:
- Multi-agent coordination with Principal Investigator and domain experts
- Vector database for memory management and context retrieval
- Knowledge repository for validated findings
- Scientific critique and quality control
- Legacy support for original atomic components
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

# Import the new multi-agent framework
from multi_agent_framework import MultiAgentResearchFramework

# Import legacy components for backward compatibility
from manuscript_drafter import draft as draft_manuscript
from literature_retriever import LiteratureRetriever
from critic import Critic
from results_visualizer import visualize as visualize_results
from specialist_registry import SpecialistRegistry
from experiments.experiment import ExperimentRunner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIPoweredResearchFramework(MultiAgentResearchFramework):
    """
    AI-Powered Research Framework that inherits from MultiAgentResearchFramework.
    
    This class provides backward compatibility while leveraging the new multi-agent
    architecture with vector database integration and intelligent agent coordination.
    
    The framework now operates as a virtual lab of AI "scientists" that collaborate
    on interdisciplinary research problems, with:
    
    - Principal Investigator (PI) Agent coordinating research
    - Agent Marketplace with domain experts (Ophthalmology, Psychology, Neuroscience, etc.)
    - Scientific Critic Agent for quality control
    - Vector Database for memory management and context retrieval
    - Knowledge Repository for validated findings
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AI-Powered Research Framework.
        
        Args:
            config: Optional configuration dictionary. If None, loads default config.
        """
        # Load legacy configuration format
        legacy_config = self._load_legacy_config(config)
        
        # Initialize the multi-agent framework
        super().__init__(legacy_config)
        
        # Initialize legacy components for backward compatibility
        self._init_legacy_compatibility()
        
        logger.info("AI-Powered Research Framework (Multi-Agent) initialized successfully")
    
    def _load_legacy_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert legacy config format to new multi-agent config."""
        legacy_defaults = {
            'experiment_db_path': 'experiments/experiments.db',
            'output_dir': 'output',
            'manuscript_dir': 'manuscripts',
            'visualization_dir': 'visualizations',
            'literature_api_url': None,
            'literature_api_key': None,
            'max_literature_results': 10,
            'auto_visualize': True,
            'auto_critique': True
        }
        
        if config:
            legacy_defaults.update(config)
            
        return legacy_defaults
    
    def _init_legacy_compatibility(self):
        """Initialize legacy components for backward compatibility."""
        # These are already initialized in the parent class, just expose them
        # for backward compatibility
        
        # Register legacy specialists in the specialist registry
        self.specialist_registry = SpecialistRegistry()
        self.specialist_registry.register('experiment_runner', self.run_experiment)
        self.specialist_registry.register('literature_retriever', self.retrieve_literature)
        self.specialist_registry.register('manuscript_drafter', self.draft_manuscript)
        self.specialist_registry.register('critic', self.critique_output)
        self.specialist_registry.register('visualizer', self.visualize_results)
        
        logger.info("Legacy compatibility components initialized")
    
    def critique_output(self, output: str) -> Dict[str, Union[str, List[str], int]]:
        """
        Legacy critique method using both traditional and multi-agent critics.
        
        Args:
            output: Text output to critique
            
        Returns:
            Dictionary containing critique results
        """
        # Use the new scientific critic agent for more comprehensive analysis
        multi_agent_critique = self.scientific_critic.critique_research_output(
            output_content=output,
            output_type="general"
        )
        
        # Convert to legacy format for backward compatibility
        return {
            'overall_score': multi_agent_critique['overall_score'],
            'recommendations': multi_agent_critique['recommendations'],
            'critical_issues': multi_agent_critique['critical_issues'],
            'detailed_analysis': multi_agent_critique
        }
    
    def run_complete_workflow(self, experiment_params: Dict[str, Any],
                            manuscript_context: Dict[str, Any],
                            literature_query: Optional[str] = None) -> Dict[str, Any]:
        """
        Legacy method: Run a complete research workflow using multi-agent system.
        
        Args:
            experiment_params: Parameters for the experiment
            manuscript_context: Context for manuscript generation
            literature_query: Optional query for literature retrieval
            
        Returns:
            Dictionary containing all workflow outputs
        """
        logger.info("Starting legacy workflow using multi-agent system")
        
        # Construct research question from experiment and manuscript context
        research_question = f"""
        Research Objective: {manuscript_context.get('objective', 'Conduct research experiment')}
        
        Experimental Setup:
        {json.dumps(experiment_params, indent=2)}
        
        Research Context:
        Methods: {manuscript_context.get('methods', 'Not specified')}
        Expected Conclusion: {manuscript_context.get('conclusion', 'To be determined')}
        """
        
        if literature_query:
            research_question += f"\n\nLiterature Focus: {literature_query}"
        
        # Use multi-agent research coordination
        research_session = self.conduct_research(
            research_question=research_question,
            context={
                'experiment_params': experiment_params,
                'manuscript_context': manuscript_context,
                'literature_query': literature_query,
                'workflow_type': 'legacy_complete_workflow'
            }
        )
        
        # Convert to legacy format
        workflow_results = {
            'workflow_id': research_session['session_id'],
            'status': research_session['status']
        }
        
        if research_session['status'] == 'completed':
            # Run the actual experiment
            experiment_results = self.run_experiment(experiment_params)
            workflow_results['experiment'] = experiment_results
            
            # Get literature if requested
            if literature_query:
                literature = self.retrieve_literature(literature_query)
                workflow_results['literature'] = literature
                
                # Integrate literature into manuscript context
                if literature:
                    manuscript_context.setdefault('references', []).extend(
                        [self._format_literature_reference(lit) for lit in literature[:3]]
                    )
            
            # Generate manuscript
            manuscript = self.draft_manuscript([experiment_results], manuscript_context)
            manuscript_path = self._save_manuscript(manuscript, workflow_results['workflow_id'])
            workflow_results['manuscript'] = {
                'content': manuscript,
                'path': manuscript_path
            }
            
            # Critique manuscript
            critique = self.critique_output(manuscript)
            workflow_results['critique'] = critique
            
            # Add multi-agent research insights
            workflow_results['multi_agent_research'] = research_session['synthesis']
            workflow_results['agent_collaboration'] = research_session['collaboration_results']
            
        else:
            workflow_results['error'] = research_session.get('error', 'Unknown error')
            
        return workflow_results
    
    def get_specialist(self, role: str):
        """Get a registered specialist by role."""
        return self.specialist_registry.get(role)
    
    def list_specialists(self) -> List[str]:
        """List all registered specialist roles."""
        return self.specialist_registry.list_roles()
    
    def save_workflow_config(self, config: Dict[str, Any], name: str) -> str:
        """
        Save a workflow configuration for reuse.
        
        Args:
            config: Configuration dictionary
            name: Name for the saved configuration
            
        Returns:
            Path to saved configuration file
        """
        config_dir = Path(self.config['output_dir']) / 'configs'
        config_dir.mkdir(exist_ok=True)
        
        config_path = config_dir / f"{name}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Workflow configuration saved: {config_path}")
        return str(config_path)
    
    def load_workflow_config(self, name: str) -> Dict[str, Any]:
        """
        Load a saved workflow configuration.
        
        Args:
            name: Name of the saved configuration
            
        Returns:
            Configuration dictionary
        """
        config_dir = Path(self.config['output_dir']) / 'configs'
        config_path = config_dir / f"{name}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration '{name}' not found")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        logger.info(f"Workflow configuration loaded: {config_path}")
        return config
    
    def _generate_visualization_path(self, experiment_id: str) -> str:
        """Generate path for experiment visualization."""
        viz_dir = Path(self.config['visualization_dir'])
        return str(viz_dir / f"experiment_{experiment_id}.png")
    
    def _save_manuscript(self, manuscript: str, workflow_id: str) -> str:
        """Save manuscript to file."""
        manuscript_dir = Path(self.config['manuscript_dir'])
        manuscript_path = manuscript_dir / f"manuscript_{workflow_id}.md"
        
        with open(manuscript_path, 'w') as f:
            f.write(manuscript)
            
        return str(manuscript_path)
    
    def _format_literature_reference(self, literature: Dict) -> Dict[str, str]:
        """Format literature result as manuscript reference."""
        return {
            'authors': literature.get('authors', 'Unknown'),
            'title': literature.get('title', 'Unknown Title'),
            'year': str(literature.get('publication_year', 'Unknown'))
        }
    
    def _generate_id(self) -> str:
        """Generate a simple ID for workflows."""
        import time
        return str(int(time.time()))


def create_framework(config: Optional[Dict[str, Any]] = None) -> AIPoweredResearchFramework:
    """
    Factory function to create an AI-Powered Research Framework instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured AIPoweredResearchFramework instance
    """
    return AIPoweredResearchFramework(config)


# Example usage
if __name__ == "__main__":
    # Create framework instance
    framework = create_framework()
    
    # Example: Multi-agent research on binocular vision and anxiety
    research_question = """
    Investigate the relationship between binocular vision dysfunction and anxiety disorders.
    Consider ophthalmological, psychological, and neurological factors.
    """
    
    # Conduct multi-agent research
    research_results = framework.conduct_research(research_question)
    print(f"Research session completed: {research_results['session_id']}")
    print(f"Status: {research_results['status']}")
    
    if research_results['status'] == 'completed':
        print(f"Synthesis: {research_results['synthesis']['synthesis_text'][:200]}...")
        print(f"Validated findings: {len(research_results['validated_findings'])}")
    
    # Example: Legacy workflow for backward compatibility
    experiment_params = {
        'study_type': 'binocular_vision_assessment',
        'participants': 50,
        'measures': ['visual_acuity', 'convergence', 'anxiety_scale'],
        'duration_weeks': 12
    }
    
    manuscript_context = {
        'objective': 'Evaluate relationship between binocular vision and anxiety',
        'methods': 'Cross-sectional study with standardized assessments',
        'conclusion': 'To be determined from data analysis'
    }
    
    # Run legacy workflow enhanced with multi-agent capabilities
    legacy_results = framework.run_complete_workflow(
        experiment_params=experiment_params,
        manuscript_context=manuscript_context,
        literature_query='binocular vision dysfunction anxiety mental health'
    )
    
    print(f"Legacy workflow completed: {legacy_results['workflow_id']}")
    print(f"Experiment ID: {legacy_results['experiment']['experiment_id']}")
    print(f"Manuscript saved to: {legacy_results['manuscript']['path']}")
    print(f"Critique score: {legacy_results['critique']['overall_score']}/100")