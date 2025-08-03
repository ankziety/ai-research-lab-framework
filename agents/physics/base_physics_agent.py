"""
Base Physics Agent - Abstract base class for physics-specific agents.

This module provides the foundational abstract base class that all physics agents
inherit from, establishing common interfaces and physics-specific capabilities.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
from enum import Enum

from ..base_agent import BaseAgent

logger = logging.getLogger(__name__)


class PhysicsScale(Enum):
    """Physical scales for cross-scale phenomena handling."""
    QUANTUM = "quantum"           # Sub-atomic scale
    ATOMIC = "atomic"             # Atomic scale  
    MOLECULAR = "molecular"       # Molecular scale
    NANO = "nano"                 # Nanoscale
    MICRO = "micro"               # Microscale
    MESO = "meso"                 # Mesoscale
    MACRO = "macro"               # Macroscale
    PLANETARY = "planetary"       # Planetary scale
    STELLAR = "stellar"           # Stellar scale
    GALACTIC = "galactic"         # Galactic scale
    COSMIC = "cosmic"             # Cosmological scale


class PhysicsMethodology(Enum):
    """Physics research methodologies."""
    THEORETICAL = "theoretical"               # Theoretical analysis
    COMPUTATIONAL = "computational"           # Computational modeling
    EXPERIMENTAL = "experimental"             # Experimental investigation
    OBSERVATIONAL = "observational"           # Observational studies
    PHENOMENOLOGICAL = "phenomenological"     # Phenomenological modeling


class BasePhysicsAgent(BaseAgent, ABC):
    """
    Abstract base class for physics-specific agents.
    
    Provides common physics-specific functionality while maintaining
    compatibility with the existing agent framework.
    """
    
    def __init__(self, agent_id: str, role: str, expertise: List[str], 
                 model_config: Optional[Dict[str, Any]] = None, 
                 cost_manager=None):
        """
        Initialize base physics agent.
        
        Args:
            agent_id: Unique identifier for the agent
            role: Agent's role (e.g., "Quantum Physics Expert")
            expertise: List of expertise domains
            model_config: Configuration for the underlying LLM
            cost_manager: Optional cost manager for tracking API usage
        """
        super().__init__(agent_id, role, expertise, model_config, cost_manager)
        
        # Physics-specific attributes
        self.physics_domain = self._get_physics_domain()
        self.physics_scales = self._get_relevant_scales()
        self.physics_methodologies = self._get_preferred_methodologies()
        self.physics_tools_cache = {}
        self.physics_knowledge_base = {}
        
        # Performance tracking for physics tasks
        self.physics_metrics = {
            'equations_solved': 0,
            'simulations_run': 0,
            'experiments_designed': 0,
            'theories_validated': 0,
            'discoveries_made': 0
        }
        
        logger.info(f"Physics Agent {self.agent_id} ({self.physics_domain}) initialized")
    
    @abstractmethod
    def _get_physics_domain(self) -> str:
        """
        Get the specific physics domain for this agent.
        
        Returns:
            Physics domain identifier (e.g., 'quantum_physics')
        """
        pass
    
    @abstractmethod
    def _get_relevant_scales(self) -> List[PhysicsScale]:
        """
        Get the physical scales relevant to this agent's domain.
        
        Returns:
            List of relevant physical scales
        """
        pass
    
    @abstractmethod
    def _get_preferred_methodologies(self) -> List[PhysicsMethodology]:
        """
        Get the preferred research methodologies for this agent.
        
        Returns:
            List of preferred methodologies
        """
        pass
    
    def discover_available_tools(self, research_question: str) -> List[Dict[str, Any]]:
        """
        Discover physics tools available for research.
        
        Args:
            research_question: Description of the research question
            
        Returns:
            List of available physics tools with confidence scores
        """
        # Cache tools for performance
        cache_key = f"{self.physics_domain}_{hash(research_question)}"
        if cache_key in self.physics_tools_cache:
            return self.physics_tools_cache[cache_key]
        
        # Get base tools from parent class
        base_tools = super().discover_available_tools(research_question)
        
        # Add physics-specific tool discovery
        physics_tools = self._discover_physics_specific_tools(research_question)
        
        # Combine and score tools
        all_tools = base_tools + physics_tools
        
        # Apply physics-specific scoring
        scored_tools = self._score_tools_for_physics(all_tools, research_question)
        
        # Cache results
        self.physics_tools_cache[cache_key] = scored_tools
        
        logger.info(f"Physics agent {self.agent_id} discovered {len(scored_tools)} tools")
        return scored_tools
    
    def optimize_tool_usage(self, research_question: str, 
                           available_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize tool selection for physics research.
        
        Args:
            research_question: Description of the research question
            available_tools: List of available tools
            
        Returns:
            Optimized list of tools with usage recommendations
        """
        # Apply base optimization
        base_optimized = super().optimize_tool_usage(research_question, available_tools)
        
        # Apply physics-specific optimization
        physics_optimized = self._optimize_physics_tools(research_question, base_optimized)
        
        # Add physics methodology considerations
        methodology_optimized = self._optimize_by_methodology(
            research_question, physics_optimized
        )
        
        # Add scale considerations
        scale_optimized = self._optimize_by_scale(research_question, methodology_optimized)
        
        logger.info(f"Physics agent {self.agent_id} optimized {len(scale_optimized)} tools")
        return scale_optimized
    
    def request_tool(self, tool_id: str, context: Dict[str, Any]) -> Optional[Any]:
        """
        Request tool access with physics-specific context.
        
        Args:
            tool_id: ID of the requested tool
            context: Execution context for validation
            
        Returns:
            Tool instance if accessible, None if not available
        """
        # Add physics-specific context
        physics_context = {
            **context,
            'physics_domain': self.physics_domain,
            'physics_scales': [scale.value for scale in self.physics_scales],
            'physics_methodologies': [method.value for method in self.physics_methodologies],
            'agent_physics_expertise': self.expertise
        }
        
        # Request tool with enhanced context
        tool = super().request_tool(tool_id, physics_context)
        
        if tool:
            # Apply physics-specific tool configuration
            self._configure_physics_tool(tool, physics_context)
        
        return tool
    
    def generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        Generate physics-specific response to prompts.
        
        Args:
            prompt: The input prompt or question
            context: Additional context information
            
        Returns:
            Generated response string with physics expertise
        """
        # Enhance prompt with physics-specific information
        physics_prompt = self._enhance_prompt_with_physics(prompt, context)
        
        # Add physics knowledge to context
        physics_context = {
            **context,
            'physics_domain': self.physics_domain,
            'physics_expertise': self.expertise,
            'physics_methodologies': [m.value for m in self.physics_methodologies],
            'physics_scales': [s.value for s in self.physics_scales]
        }
        
        # Generate response using parent method with enhanced context
        response = super().generate_response(physics_prompt, physics_context)
        
        # Post-process response with physics-specific formatting
        formatted_response = self._format_physics_response(response, context)
        
        return formatted_response
    
    def assess_physics_complexity(self, research_question: str) -> Dict[str, Any]:
        """
        Assess the physics complexity of a research question.
        
        Args:
            research_question: Research question to assess
            
        Returns:
            Complexity assessment including scales, methodologies, and difficulty
        """
        complexity = {
            'overall_score': 0.0,
            'scale_complexity': self._assess_scale_complexity(research_question),
            'methodology_complexity': self._assess_methodology_complexity(research_question),
            'domain_complexity': self._assess_domain_complexity(research_question),
            'theoretical_depth': 0.0,
            'computational_requirements': 0.0,
            'experimental_difficulty': 0.0
        }
        
        # Calculate overall complexity score
        complexity['overall_score'] = (
            complexity['scale_complexity'] * 0.25 +
            complexity['methodology_complexity'] * 0.25 +
            complexity['domain_complexity'] * 0.25 +
            complexity['theoretical_depth'] * 0.25
        )
        
        return complexity
    
    def solve_physics_equation(self, equation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve physics equations using appropriate methods.
        
        Args:
            equation: Mathematical equation or system description
            parameters: Parameters and boundary conditions
            
        Returns:
            Solution results including methods used and validation
        """
        result = {
            'success': False,
            'solution': None,
            'method': 'unknown',
            'validation': {},
            'uncertainty': 0.0,
            'computational_cost': 0.0
        }
        
        try:
            # Select appropriate solution method
            method = self._select_solution_method(equation, parameters)
            
            # Solve using selected method
            solution = self._apply_solution_method(equation, parameters, method)
            
            # Validate solution
            validation = self._validate_physics_solution(solution, equation, parameters)
            
            result.update({
                'success': True,
                'solution': solution,
                'method': method,
                'validation': validation,
                'uncertainty': validation.get('uncertainty', 0.0)
            })
            
            # Update metrics
            self.physics_metrics['equations_solved'] += 1
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Equation solving failed for agent {self.agent_id}: {e}")
        
        return result
    
    def design_physics_experiment(self, hypothesis: str, 
                                 constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Design physics experiments based on hypothesis and constraints.
        
        Args:
            hypothesis: Scientific hypothesis to test
            constraints: Experimental constraints (budget, time, equipment)
            
        Returns:
            Experimental design including methodology and expected outcomes
        """
        design = {
            'success': False,
            'methodology': 'unknown',
            'equipment_needed': [],
            'procedures': [],
            'expected_outcomes': [],
            'uncertainty_analysis': {},
            'statistical_power': 0.0,
            'estimated_cost': 0.0,
            'estimated_duration': 0.0
        }
        
        try:
            # Analyze hypothesis for experimental requirements
            requirements = self._analyze_experimental_requirements(hypothesis)
            
            # Design experimental methodology
            methodology = self._design_experimental_methodology(
                hypothesis, requirements, constraints
            )
            
            # Plan experimental procedures
            procedures = self._plan_experimental_procedures(methodology, constraints)
            
            # Perform uncertainty analysis
            uncertainty_analysis = self._perform_uncertainty_analysis(
                methodology, procedures
            )
            
            design.update({
                'success': True,
                'methodology': methodology,
                'procedures': procedures,
                'uncertainty_analysis': uncertainty_analysis,
                'statistical_power': uncertainty_analysis.get('statistical_power', 0.0)
            })
            
            # Update metrics
            self.physics_metrics['experiments_designed'] += 1
            
        except Exception as e:
            design['error'] = str(e)
            logger.error(f"Experiment design failed for agent {self.agent_id}: {e}")
        
        return design
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert agent to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation including physics-specific data
        """
        base_dict = super().to_dict()
        
        physics_dict = {
            **base_dict,
            'physics_domain': self.physics_domain,
            'physics_scales': [scale.value for scale in self.physics_scales],
            'physics_methodologies': [method.value for method in self.physics_methodologies],
            'physics_metrics': self.physics_metrics.copy(),
            'agent_type': 'BasePhysicsAgent',
            'physics_version': '1.0.0'
        }
        
        return physics_dict
    
    # Private helper methods
    
    def _discover_physics_specific_tools(self, research_question: str) -> List[Dict[str, Any]]:
        """Discover tools specific to physics research."""
        physics_tools = []
        
        # Physics simulation tools
        if any(keyword in research_question.lower() for keyword in 
               ['simulate', 'model', 'calculate', 'compute']):
            physics_tools.append({
                'tool_id': 'physics_simulator',
                'name': 'Physics Simulation Engine',
                'description': 'Advanced physics simulation and modeling tool',
                'capabilities': ['numerical_simulation', 'equation_solving', 'visualization'],
                'confidence': 0.9,
                'physics_specific': True
            })
        
        # Mathematical analysis tools
        if any(keyword in research_question.lower() for keyword in 
               ['equation', 'solve', 'analyze', 'mathematical']):
            physics_tools.append({
                'tool_id': 'math_analyzer',
                'name': 'Mathematical Analysis Tool',
                'description': 'Advanced mathematical analysis for physics',
                'capabilities': ['symbolic_math', 'numerical_analysis', 'differential_equations'],
                'confidence': 0.85,
                'physics_specific': True
            })
        
        # Experimental design tools
        if any(keyword in research_question.lower() for keyword in 
               ['experiment', 'measure', 'test', 'validate']):
            physics_tools.append({
                'tool_id': 'experiment_designer',
                'name': 'Physics Experiment Designer',
                'description': 'Design and plan physics experiments',
                'capabilities': ['experimental_design', 'uncertainty_analysis', 'statistical_planning'],
                'confidence': 0.8,
                'physics_specific': True
            })
        
        return physics_tools
    
    def _score_tools_for_physics(self, tools: List[Dict[str, Any]], 
                                research_question: str) -> List[Dict[str, Any]]:
        """Apply physics-specific scoring to tools."""
        scored_tools = []
        
        for tool in tools:
            score = tool.get('confidence', 0.0)
            
            # Boost physics-specific tools
            if tool.get('physics_specific', False):
                score += 0.2
            
            # Score based on domain relevance
            if self.physics_domain in tool.get('capabilities', []):
                score += 0.15
            
            # Score based on scale relevance
            tool_scales = tool.get('scales', [])
            scale_overlap = len(set(tool_scales) & set([s.value for s in self.physics_scales]))
            if scale_overlap > 0:
                score += 0.1 * scale_overlap
            
            tool['confidence'] = min(1.0, score)
            scored_tools.append(tool)
        
        # Sort by confidence score
        scored_tools.sort(key=lambda x: x['confidence'], reverse=True)
        
        return scored_tools
    
    def _optimize_physics_tools(self, research_question: str, 
                               tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply physics-specific optimization to tools."""
        optimized = []
        
        for tool in tools:
            optimization_score = tool.get('optimization_score', 0.0)
            
            # Factor in physics domain compatibility
            if tool.get('physics_specific', False):
                optimization_score += 0.3
            
            # Factor in methodology compatibility
            tool_methods = tool.get('methodologies', [])
            method_overlap = len(set(tool_methods) & 
                               set([m.value for m in self.physics_methodologies]))
            if method_overlap > 0:
                optimization_score += 0.2 * method_overlap
            
            tool['optimization_score'] = optimization_score
            optimized.append(tool)
        
        return optimized
    
    def _optimize_by_methodology(self, research_question: str, 
                                tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize tools based on physics methodology requirements."""
        # Analyze question for methodology requirements
        methodology_requirements = self._extract_methodology_requirements(research_question)
        
        for tool in tools:
            # Score tools based on methodology compatibility
            tool_methodologies = tool.get('methodologies', [])
            compatibility_score = 0.0
            
            for req_method in methodology_requirements:
                if req_method in tool_methodologies:
                    compatibility_score += 1.0
            
            if methodology_requirements:
                compatibility_score /= len(methodology_requirements)
            
            tool['methodology_score'] = compatibility_score
        
        return tools
    
    def _optimize_by_scale(self, research_question: str, 
                          tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize tools based on physics scale requirements."""
        # Analyze question for scale requirements
        scale_requirements = self._extract_scale_requirements(research_question)
        
        for tool in tools:
            # Score tools based on scale compatibility
            tool_scales = tool.get('scales', [])
            scale_score = 0.0
            
            for req_scale in scale_requirements:
                if req_scale in tool_scales:
                    scale_score += 1.0
            
            if scale_requirements:
                scale_score /= len(scale_requirements)
            
            tool['scale_score'] = scale_score
        
        return tools
    
    def _configure_physics_tool(self, tool: Any, context: Dict[str, Any]) -> None:
        """Configure tool with physics-specific settings."""
        if hasattr(tool, 'set_physics_context'):
            tool.set_physics_context(context)
        
        if hasattr(tool, 'set_domain'):
            tool.set_domain(self.physics_domain)
    
    def _enhance_prompt_with_physics(self, prompt: str, context: Dict[str, Any]) -> str:
        """Enhance prompt with physics-specific information."""
        physics_enhancement = f"""
        You are a {self.role} with deep expertise in {', '.join(self.expertise)}.
        Your physics domain is {self.physics_domain}.
        
        Relevant physical scales: {', '.join([scale.value for scale in self.physics_scales])}
        Preferred methodologies: {', '.join([method.value for method in self.physics_methodologies])}
        
        When responding to: {prompt}
        
        Please provide a comprehensive physics analysis including:
        1. **Physical Principles**: Relevant fundamental physics principles
        2. **Mathematical Framework**: Key equations and mathematical formulations
        3. **Scale Considerations**: Multi-scale physics effects if applicable
        4. **Methodology**: Appropriate research/analysis methodology
        5. **Validation**: How to validate or test the analysis
        
        Use proper physics notation and terminology. Be rigorous but accessible.
        """
        
        return physics_enhancement
    
    def _format_physics_response(self, response: str, context: Dict[str, Any]) -> str:
        """Format response with physics-specific formatting."""
        # Add physics notation and formatting
        formatted = f"""
## Physics Analysis by {self.role}

{response}

---
**Domain**: {self.physics_domain}  
**Scale(s)**: {', '.join([scale.value for scale in self.physics_scales])}  
**Methodology**: {', '.join([method.value for method in self.physics_methodologies])}  

*This analysis is provided by a specialized physics AI agent with expertise in {', '.join(self.expertise)}.*
        """
        
        return formatted.strip()
    
    # Abstract methods for complexity assessment (to be implemented by subclasses)
    
    def _assess_scale_complexity(self, research_question: str) -> float:
        """Assess complexity based on physical scales involved."""
        return 0.5  # Default implementation
    
    def _assess_methodology_complexity(self, research_question: str) -> float:
        """Assess complexity based on required methodologies."""
        return 0.5  # Default implementation
    
    def _assess_domain_complexity(self, research_question: str) -> float:
        """Assess complexity based on domain-specific factors."""
        return 0.5  # Default implementation
    
    # Abstract methods for equation solving (to be implemented by subclasses)
    
    def _select_solution_method(self, equation: str, parameters: Dict[str, Any]) -> str:
        """Select appropriate method for solving physics equation."""
        return 'numerical'  # Default implementation
    
    def _apply_solution_method(self, equation: str, parameters: Dict[str, Any], 
                              method: str) -> Any:
        """Apply selected solution method."""
        return {'result': 'placeholder_solution'}  # Default implementation
    
    def _validate_physics_solution(self, solution: Any, equation: str, 
                                  parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate physics solution."""
        return {'uncertainty': 0.1, 'confidence': 0.9}  # Default implementation
    
    # Abstract methods for experimental design (to be implemented by subclasses)
    
    def _analyze_experimental_requirements(self, hypothesis: str) -> Dict[str, Any]:
        """Analyze hypothesis for experimental requirements."""
        return {'type': 'measurement', 'precision': 'high'}  # Default implementation
    
    def _design_experimental_methodology(self, hypothesis: str, 
                                        requirements: Dict[str, Any],
                                        constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Design experimental methodology."""
        return {'approach': 'controlled_experiment'}  # Default implementation
    
    def _plan_experimental_procedures(self, methodology: Dict[str, Any], 
                                     constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan experimental procedures."""
        return [{'step': 1, 'action': 'setup_equipment'}]  # Default implementation
    
    def _perform_uncertainty_analysis(self, methodology: Dict[str, Any], 
                                     procedures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform uncertainty analysis for experiment."""
        return {'statistical_power': 0.8, 'uncertainty': 0.05}  # Default implementation
    
    # Helper methods for prompt analysis
    
    def _extract_methodology_requirements(self, research_question: str) -> List[str]:
        """Extract methodology requirements from research question."""
        methodology_keywords = {
            'theoretical': ['theory', 'theoretical', 'analytical', 'mathematical'],
            'computational': ['simulation', 'computational', 'numerical', 'modeling'],
            'experimental': ['experiment', 'experimental', 'measurement', 'test'],
            'observational': ['observation', 'observational', 'survey', 'monitoring']
        }
        
        requirements = []
        question_lower = research_question.lower()
        
        for methodology, keywords in methodology_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                requirements.append(methodology)
        
        return requirements
    
    def _extract_scale_requirements(self, research_question: str) -> List[str]:
        """Extract scale requirements from research question."""
        scale_keywords = {
            'quantum': ['quantum', 'qubit', 'photon', 'electron'],
            'atomic': ['atom', 'atomic', 'molecular'],
            'nano': ['nano', 'nanoscale', 'nanometer'],
            'micro': ['micro', 'microscale', 'micrometer'],
            'macro': ['macro', 'macroscopic', 'bulk'],
            'planetary': ['planet', 'planetary', 'earth'],
            'stellar': ['star', 'stellar', 'solar'],
            'galactic': ['galaxy', 'galactic', 'cosmic'],
            'cosmic': ['universe', 'cosmological', 'cosmos']
        }
        
        requirements = []
        question_lower = research_question.lower()
        
        for scale, keywords in scale_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                requirements.append(scale)
        
        return requirements