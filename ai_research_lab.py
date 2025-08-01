#!/usr/bin/env python3
"""
AI Research Lab Framework

A comprehensive framework for AI research workflows that integrates experiment 
execution, literature retrieval, manuscript drafting, result visualization, 
and research critique capabilities.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

# Import all the atomic components
from manuscript_drafter import draft as draft_manuscript
from literature_retriever import LiteratureRetriever
from critic import Critic
from results_visualizer import visualize as visualize_results
from specialist_registry import SpecialistRegistry
from experiments.experiment import ExperimentRunner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIResearchLabFramework:
    """
    Main framework class that orchestrates all AI research lab components.
    
    This class provides a unified interface for conducting AI research workflows,
    from running experiments to generating final manuscript drafts with critique.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AI Research Lab Framework.
        
        Args:
            config: Optional configuration dictionary. If None, loads default config.
        """
        self.config = self._load_config(config)
        
        # Initialize all components
        self.experiment_runner = ExperimentRunner(
            db_path=self.config.get('experiment_db_path')
        )
        self.literature_retriever = LiteratureRetriever(
            api_base_url=self.config.get('literature_api_url'),
            api_key=self.config.get('literature_api_key')
        )
        self.critic = Critic()
        self.specialist_registry = SpecialistRegistry()
        
        # Setup output directories
        self._setup_directories()
        
        # Register default specialists
        self._register_default_specialists()
        
        logger.info("AI Research Lab Framework initialized successfully")
    
    def _load_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load configuration from provided config or defaults."""
        default_config = {
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
            default_config.update(config)
            
        return default_config
    
    def _setup_directories(self) -> None:
        """Create necessary output directories."""
        dirs_to_create = [
            self.config['output_dir'],
            self.config['manuscript_dir'],
            self.config['visualization_dir']
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _register_default_specialists(self) -> None:
        """Register default specialist handlers."""
        self.specialist_registry.register('experiment_runner', self.run_experiment)
        self.specialist_registry.register('literature_retriever', self.retrieve_literature)
        self.specialist_registry.register('manuscript_drafter', self.draft_manuscript)
        self.specialist_registry.register('critic', self.critique_output)
        self.specialist_registry.register('visualizer', self.visualize_results)
    
    def run_experiment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run an experiment and optionally auto-generate visualizations.
        
        Args:
            params: Experiment parameters
            
        Returns:
            Dictionary containing experiment results and metadata
        """
        logger.info(f"Running experiment with parameters: {params}")
        
        # Run the actual experiment
        results = self.experiment_runner.run_experiment(params)
        
        # Auto-visualize results if enabled
        if self.config.get('auto_visualize') and 'computed_results' in results:
            try:
                viz_path = self._generate_visualization_path(results['experiment_id'])
                self.visualize_results([results], viz_path)
                results['visualization_path'] = viz_path
                logger.info(f"Auto-generated visualization: {viz_path}")
            except Exception as e:
                logger.warning(f"Auto-visualization failed: {e}")
        
        return results
    
    def retrieve_literature(self, query: str, max_results: Optional[int] = None) -> List[Dict]:
        """
        Retrieve relevant literature for a research topic.
        
        Args:
            query: Search query string
            max_results: Maximum number of results (uses config default if None)
            
        Returns:
            List of literature records
        """
        if max_results is None:
            max_results = self.config.get('max_literature_results', 10)
            
        logger.info(f"Retrieving literature for query: '{query}'")
        
        return self.literature_retriever.search(query, max_results)
    
    def draft_manuscript(self, results: List[Dict[str, Any]], 
                        context: Dict[str, Any]) -> str:
        """
        Draft a scientific manuscript from experimental results.
        
        Args:
            results: List of experimental results
            context: Contextual information for the manuscript
            
        Returns:
            Formatted manuscript string
        """
        logger.info("Drafting manuscript from experimental results")
        
        manuscript = draft_manuscript(results, context)
        
        # Auto-critique if enabled
        if self.config.get('auto_critique'):
            try:
                critique = self.critique_output(manuscript)
                # Add critique as metadata (could be used for revision)
                logger.info(f"Auto-critique score: {critique['overall_score']}/100")
            except Exception as e:
                logger.warning(f"Auto-critique failed: {e}")
        
        return manuscript
    
    def critique_output(self, output: str) -> Dict[str, Union[str, List[str], int]]:
        """
        Critique a research output.
        
        Args:
            output: Text output to critique
            
        Returns:
            Dictionary containing critique results
        """
        logger.info("Critiquing research output")
        return self.critic.review(output)
    
    def visualize_results(self, results: List[Dict], out_path: str) -> None:
        """
        Generate visualizations from experimental results.
        
        Args:
            results: List of result dictionaries
            out_path: Output path for visualization
        """
        logger.info(f"Generating visualization: {out_path}")
        visualize_results(results, out_path)
    
    def run_complete_workflow(self, experiment_params: Dict[str, Any],
                            manuscript_context: Dict[str, Any],
                            literature_query: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a complete research workflow from experiment to manuscript.
        
        Args:
            experiment_params: Parameters for the experiment
            manuscript_context: Context for manuscript generation
            literature_query: Optional query for literature retrieval
            
        Returns:
            Dictionary containing all workflow outputs
        """
        logger.info("Starting complete research workflow")
        
        workflow_results = {
            'workflow_id': f"workflow_{self._generate_id()}",
            'status': 'running'
        }
        
        try:
            # Step 1: Run experiment
            experiment_results = self.run_experiment(experiment_params)
            workflow_results['experiment'] = experiment_results
            
            # Step 2: Retrieve literature (if query provided)
            if literature_query:
                literature = self.retrieve_literature(literature_query)
                workflow_results['literature'] = literature
                
                # Integrate literature into manuscript context
                if literature:
                    manuscript_context.setdefault('references', []).extend(
                        [self._format_literature_reference(lit) for lit in literature[:3]]
                    )
            
            # Step 3: Generate manuscript
            manuscript = self.draft_manuscript([experiment_results], manuscript_context)
            manuscript_path = self._save_manuscript(manuscript, workflow_results['workflow_id'])
            workflow_results['manuscript'] = {
                'content': manuscript,
                'path': manuscript_path
            }
            
            # Step 4: Critique manuscript
            critique = self.critique_output(manuscript)
            workflow_results['critique'] = critique
            
            workflow_results['status'] = 'completed'
            logger.info(f"Workflow completed successfully: {workflow_results['workflow_id']}")
            
        except Exception as e:
            workflow_results['status'] = 'failed'
            workflow_results['error'] = str(e)
            logger.error(f"Workflow failed: {e}")
            
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


def create_framework(config: Optional[Dict[str, Any]] = None) -> AIResearchLabFramework:
    """
    Factory function to create an AI Research Lab Framework instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured AIResearchLabFramework instance
    """
    return AIResearchLabFramework(config)


# Example usage
if __name__ == "__main__":
    # Create framework instance
    framework = create_framework()
    
    # Example workflow
    experiment_params = {
        'algorithm': 'neural_network',
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32
    }
    
    manuscript_context = {
        'objective': 'Evaluate neural network performance',
        'methods': 'Deep learning with backpropagation',
        'conclusion': 'Neural network achieved high accuracy'
    }
    
    # Run complete workflow
    results = framework.run_complete_workflow(
        experiment_params=experiment_params,
        manuscript_context=manuscript_context,
        literature_query='neural networks machine learning'
    )
    
    print(f"Workflow completed: {results['workflow_id']}")
    print(f"Experiment ID: {results['experiment']['experiment_id']}")
    print(f"Manuscript saved to: {results['manuscript']['path']}")
    print(f"Critique score: {results['critique']['overall_score']}/100")